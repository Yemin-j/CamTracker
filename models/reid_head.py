import torch
import torch.nn as nn
import torch.nn.functional as F

class ReIDHead(nn.Module):
    def __init__(self, in_channels=256, embed_dim=256, temperature=0.07):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

        # Embedding extractor (4conv1fc with GroupNorm like QDTrack)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(32, in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(32, in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.gn3 = nn.GroupNorm(32, in_channels)
        self.conv4 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.gn4 = nn.GroupNorm(32, in_channels)

        self.fc = nn.Linear(in_channels * 7 * 7, embed_dim)

    def forward(self, roi_feats):
        """
                roi_feats: (N, C, 7, 7)
                return: (N, embed_dim) normalized embeddings
                """
        x = roi_feats

        # 4 conv layers with GroupNorm and ReLU
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        x = F.relu(self.gn4(self.conv4(x)))

        # Flatten and FC
        x = x.flatten(1)  # (N, C*7*7)
        x = self.fc(x)  # (N, embed_dim)

        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x

    def loss(self, embeds_key, embeds_ref, match_matrix):
        """
        QDTrack-style multi-positive contrastive loss (Eq. 5 in paper)

        Args:
            embeds_key: (V, D) embeddings from key frame (training samples)
            embeds_ref: (K, D) embeddings from reference frame (contrastive targets)
            match_matrix: (V, K) binary matrix
                          match_matrix[i,j]=1 if positive pair, 0 if negative

        Returns:
            dict with loss_contrastive and loss_aux
        """
        V, D = embeds_key.shape
        K = embeds_ref.shape[0]

        if V == 0 or K == 0:
            device = embeds_key.device
            return dict(
                loss_contrastive=torch.tensor(0.0, device=device),
                loss_aux=torch.tensor(0.0, device=device),
                loss_reid=torch.tensor(0.0, device=device)
            )

        # Compute cosine similarity: (V, K)
        similarity = torch.matmul(embeds_key, embeds_ref.t()) / self.temperature

        # Multi-positive contrastive loss (Eq. 5)
        loss_contrastive = self._multi_positive_contrastive_loss(
            similarity, match_matrix
        )

        # Auxiliary L2 loss (Eq. 6)
        loss_aux = self._auxiliary_l2_loss(
            embeds_key, embeds_ref, match_matrix
        )

        # Total ReID loss
        loss_reid = loss_contrastive + loss_aux

        return dict(
            loss_contrastive=loss_contrastive,
            loss_aux=loss_aux,
            loss_reid=loss_reid
        )

    def _multi_positive_contrastive_loss(self, similarity, match_matrix):
        """
        Equation 5 in QDTrack paper:
        L_cont = -1/V * Σ_v [ log( Σ_k+ exp(s_vk+) / Σ_k exp(s_vk) ) ]

        Args:
            similarity: (V, K) similarity scores
            match_matrix: (V, K) binary positive/negative indicator
        """
        V, K = similarity.shape

        # For numerical stability
        similarity_exp = torch.exp(similarity)  # (V, K)

        # Denominator: sum over all K (positive + negative)
        denom = similarity_exp.sum(dim=1, keepdim=True)  # (V, 1)

        # Numerator: sum over positive pairs only
        pos_mask = match_matrix.float()  # (V, K)
        numer = (similarity_exp * pos_mask).sum(dim=1, keepdim=True)  # (V, 1)

        # Avoid log(0)
        numer = numer.clamp(min=1e-8)
        denom = denom.clamp(min=1e-8)

        # Log ratio
        loss = -torch.log(numer / denom)  # (V, 1)

        # Average over V samples
        loss = loss.mean()

        return loss

    def _auxiliary_l2_loss(self, embeds_key, embeds_ref, match_matrix,
                           pos_ratio=1.0, neg_ratio=3.0):
        """
        Equation 6 in QDTrack paper:
        L_aux = Σ c_ij * ||v_i - k_j||^2

        Sample positive and negative pairs for efficiency
        (paper samples all positive + 3x more negative)

        Args:
            embeds_key: (V, D)
            embeds_ref: (K, D)
            match_matrix: (V, K)
            pos_ratio: sample all positive pairs (1.0 = 100%)
            neg_ratio: sample 3x negative pairs relative to positive
        """
        V, D = embeds_key.shape
        K = embeds_ref.shape[0]

        # Find positive and negative pairs
        pos_indices = torch.nonzero(match_matrix > 0, as_tuple=False)  # (P, 2)
        neg_indices = torch.nonzero(match_matrix == 0, as_tuple=False)  # (N, 2)

        num_pos = pos_indices.shape[0]
        num_neg = neg_indices.shape[0]

        if num_pos == 0:
            return torch.tensor(0.0, device=embeds_key.device)

        # Sample positive pairs (use all by default)
        num_pos_sample = int(num_pos * pos_ratio)
        if num_pos_sample < num_pos:
            perm = torch.randperm(num_pos, device=embeds_key.device)[:num_pos_sample]
            pos_indices = pos_indices[perm]

        # Sample negative pairs (3x more than positive)
        num_neg_sample = min(int(num_pos_sample * neg_ratio), num_neg)
        if num_neg_sample > 0 and num_neg > 0:
            perm = torch.randperm(num_neg, device=embeds_key.device)[:num_neg_sample]
            neg_indices = neg_indices[perm]

            # Combine positive and negative
            all_indices = torch.cat([pos_indices, neg_indices], dim=0)
        else:
            all_indices = pos_indices

        # Extract embeddings for sampled pairs
        v_idx = all_indices[:, 0]
        k_idx = all_indices[:, 1]

        v_embeds = embeds_key[v_idx]  # (S, D)
        k_embeds = embeds_ref[k_idx]  # (S, D)

        # L2 distance
        l2_dist = torch.sum((v_embeds - k_embeds) ** 2, dim=1)  # (S,)

        # Average
        loss_aux = l2_dist.mean()

        return loss_aux

    def compute_similarity_matrix(self, embeds1, embeds2):
        """
        Compute cosine similarity matrix between two sets of embeddings
        Used for inference/tracking

        Args:
            embeds1: (N1, D)
            embeds2: (N2, D)

        Returns:
            similarity: (N1, N2) cosine similarity matrix
        """
        # Already L2-normalized in forward pass
        similarity = torch.matmul(embeds1, embeds2.t())
        return similarity