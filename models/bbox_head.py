import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.ops import nms


class BBoxHead(nn.Module):
    """
    BBox Head for QDTrack

    Handles:
    - Classification (person vs background)
    - Bounding box regression
    - NMS post-processing
    """

    def __init__(self, in_channels=256, fc_dim=1024, num_classes=1,
                 score_thresh=0.05, nms_thresh=0.5, max_dets=100):
        """
        Args:
            in_channels: Input feature channels (256 for FPN)
            fc_dim: Hidden dimension for FC layers
            num_classes: Number of foreground classes (1 for person-only)
            score_thresh: Score threshold for filtering detections
            nms_thresh: NMS IoU threshold
            max_dets: Maximum number of detections to keep
        """
        super().__init__()

        # Two FC layers for feature transformation
        self.fc1 = nn.Linear(in_channels * 7 * 7, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)

        # Classification head (num_classes + 1 for background)
        self.cls = nn.Linear(fc_dim, num_classes + 1)  # +1 for background class

        # Regression head (4 values per class, but typically only use one set)
        # For single-class detector, just output 4 values
        self.reg = nn.Linear(fc_dim, 4)  # Always output 4 regardless of num_classes

        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_dets = max_dets

    def forward(self, roi_feats):
        """
        Forward pass

        Args:
            roi_feats: (N, C, 7, 7) RoI-aligned features

        Returns:
            cls_scores: (N, C+1) classification logits
            bbox_preds: (N, 4) bbox regression deltas
        """
        x = roi_feats.flatten(1)  # (N, C*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        cls_scores = self.cls(x)  # (N, C+1)
        bbox_preds = self.reg(x)  # (N, 4)

        return cls_scores, bbox_preds

    def loss(self, cls_scores, bbox_preds, gt_labels, gt_deltas):
        """
        Compute classification and regression losses

        Args:
            cls_scores: (N, C+1) classification logits
            bbox_preds: (N, 4) predicted deltas
            gt_labels: (N,) ground truth labels
                       - 0: background
                       - 1: person (foreground)
            gt_deltas: (N, 4) ground truth regression targets
                       - Only meaningful for positive samples (label=1)
                       - Zero for negative samples (label=0)

        Returns:
            dict with loss_cls and loss_reg
        """
        # Classification loss (Cross-Entropy)
        # Handles both positive (1) and negative (0) samples
        loss_cls = F.cross_entropy(cls_scores, gt_labels, reduction='mean')

        # Regression loss (Smooth L1)
        # Only compute for positive samples
        pos_mask = gt_labels > 0

        if pos_mask.sum() > 0:
            bbox_preds_pos = bbox_preds[pos_mask]
            gt_deltas_pos = gt_deltas[pos_mask]
            loss_reg = F.smooth_l1_loss(bbox_preds_pos, gt_deltas_pos, reduction='mean')
        else:
            # No positive samples - zero loss
            loss_reg = torch.tensor(0.0, device=cls_scores.device, requires_grad=True)

        return dict(
            loss_cls=loss_cls,
            loss_reg=loss_reg
        )

    @torch.no_grad()
    def postprocess(self, cls_scores, bbox_preds, proposals, img):
        """
        Post-process detections: decode boxes, apply NMS

        Args:
            cls_scores: (N, C+1) classification logits
            bbox_preds: (N, 4) bbox regression deltas
            proposals: (N, 4) proposal boxes in xyxy format
            img: (3, H, W) input image (not used, for compatibility)

        Returns:
            final_boxes: (K, 4) final detected boxes in xyxy format
            final_scores: (K,) confidence scores
            final_labels: (K,) class labels (all 1 for person)
            keep_idx_original: (K,) indices mapping to original proposals
        """
        device = proposals.device
        N = proposals.shape[0]

        # ----- 1. Get scores for foreground class -----
        probs = F.softmax(cls_scores, dim=1)  # (N, C+1)
        scores = probs[:, 1]  # Person class (class=1)

        # ----- 2. Score threshold filtering -----
        keep_mask = scores > self.score_thresh

        if keep_mask.sum() == 0:
            # No detections above threshold
            empty_boxes = torch.empty((0, 4), device=device, dtype=torch.float32)
            empty_scores = torch.empty((0,), device=device, dtype=torch.float32)
            empty_labels = torch.empty((0,), dtype=torch.long, device=device)
            empty_keep = torch.empty((0,), dtype=torch.long, device=device)
            return empty_boxes, empty_scores, empty_labels, empty_keep

        # Get indices of kept proposals
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()

        scores_filtered = scores[keep_mask]
        proposals_filtered = proposals[keep_mask]
        deltas_filtered = bbox_preds[keep_mask]

        # ----- 3. Decode boxes: apply deltas to proposals -----
        decoded_boxes = self._decode_boxes(proposals_filtered, deltas_filtered)

        # ----- 4. Clip boxes to image boundaries -----
        # Note: img shape is (C, H, W)
        if img is not None and len(img.shape) == 3:
            _, img_h, img_w = img.shape
            decoded_boxes[:, 0].clamp_(min=0, max=img_w - 1)
            decoded_boxes[:, 1].clamp_(min=0, max=img_h - 1)
            decoded_boxes[:, 2].clamp_(min=0, max=img_w - 1)
            decoded_boxes[:, 3].clamp_(min=0, max=img_h - 1)

        # ----- 5. NMS -----
        nms_keep_idx = nms(decoded_boxes, scores_filtered, self.nms_thresh)
        nms_keep_idx = nms_keep_idx[: self.max_dets]

        # Map back to original proposal indices
        keep_idx_original = keep_indices[nms_keep_idx]

        # Final outputs
        final_boxes = decoded_boxes[nms_keep_idx]
        final_scores = scores_filtered[nms_keep_idx]
        final_labels = torch.ones_like(final_scores, dtype=torch.long)

        return final_boxes, final_scores, final_labels, keep_idx_original

    @staticmethod
    def _decode_boxes(proposals, deltas):
        """
        Decode bounding boxes from proposals and deltas (Faster R-CNN style)

        Args:
            proposals: (N, 4) boxes in xyxy format
            deltas: (N, 4) regression deltas [dx, dy, dw, dh]

        Returns:
            decoded: (N, 4) decoded boxes in xyxy format
        """
        # Convert proposals to center format
        widths = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
        heights = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
        ctr_x = proposals[:, 0] + 0.5 * widths
        ctr_y = proposals[:, 1] + 0.5 * heights

        # Extract deltas
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        # Apply deltas
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = widths * torch.exp(dw.clamp(max=10))  # Clamp for numerical stability
        pred_h = heights * torch.exp(dh.clamp(max=10))

        # Convert back to xyxy format
        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        decoded = torch.stack([x1, y1, x2, y2], dim=1)
        return decoded