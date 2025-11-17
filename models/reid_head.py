import torch.nn as nn
import torch.nn.functional as F

class ReIDHead(nn.Module):
    def __init__(self, in_channels=256, embed_dim=256, num_ids=1000):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, embed_dim)
        # ID 분류용 classifier
        self.id_classifier = nn.Linear(embed_dim, num_ids)

    def forward(self, roi_feats):
        x = roi_feats.flatten(1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.normalize(self.fc2(x), dim=1)
        return x

    def loss(self, embeds, gt_ids):
        """
        embeds: (N, D)
        gt_ids: (N,) 0..num_ids-1 (미리 매핑된 ID 인덱스)
        """
        logits = self.id_classifier(embeds)  # (N, num_ids)
        loss_id = F.cross_entropy(logits, gt_ids)
        return dict(loss_reid=loss_id)