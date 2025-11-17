import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.ops import nms

class BBoxHead(nn.Module):
    def __init__(self, in_channels=256, fc_dim=1024, num_classes=1,
                 score_thresh=0.05, nms_thresh=0.5, max_dets=100):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.cls = nn.Linear(fc_dim, num_classes + 1)  # + background
        self.reg = nn.Linear(fc_dim, num_classes * 4)

        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_dets = max_dets

    def forward(self, roi_feats):
        x = roi_feats.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_scores = self.cls(x)              # (N, C+1)
        bbox_preds = self.reg(x)              # (N, 4)
        return cls_scores, bbox_preds

    def loss(self, cls_scores, bbox_preds, gt_labels, gt_deltas):
        """
        cls_scores: (N, C+1)
        bbox_preds: (N, 4)
        gt_labels:  (N,)   (1: person, 0: background) - 여기선 GT만 쓰므로 모두 1로 가정
        gt_deltas:  (N, 4) regression target (여기선 0으로 두고 box 유지 학습)
        """
        # classification loss
        # GT는 모두 class 1 (person)이라고 가정
        loss_cls = F.cross_entropy(cls_scores, gt_labels)

        # bbox regression (smooth L1)
        loss_reg = F.smooth_l1_loss(bbox_preds, gt_deltas, reduction='mean')

        return dict(
            loss_cls=loss_cls,
            loss_reg=loss_reg
        )

    @torch.no_grad()
    def postprocess(self, cls_scores, bbox_preds, proposals, img):
        """
        proposals: (N,4) xyxy
        cls_scores: (N, C+1)
        bbox_preds: (N, C*4)
        """

        # ----- 1. Softmax -----
        probs = F.softmax(cls_scores, dim=1)          # (N, C+1)
        scores = probs[:, 1]                          # person만 (class=1 assumed)

        # ----- 2. score threshold -----
        keep = scores > self.score_thresh
        if keep.sum() == 0:
            return torch.empty((0,4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

        scores = scores[keep]
        proposals = proposals[keep]
        deltas = bbox_preds[keep]                     # (N,4)

        # ----- 3. Decode box: apply deltas to proposals -----
        # proposals: xyxy
        widths  = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        ctr_x   = proposals[:, 0] + 0.5 * widths
        ctr_y   = proposals[:, 1] + 0.5 * heights

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = widths * torch.exp(dw)
        pred_h = heights * torch.exp(dh)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        decoded = torch.stack([x1, y1, x2, y2], dim=1)

        # ----- 4. NMS -----
        keep_idx = nms(decoded, scores, self.nms_thresh)
        keep_idx = keep_idx[: self.max_dets]

        final_boxes = decoded[keep_idx]
        final_scores = scores[keep_idx]
        final_labels = torch.ones_like(final_scores, dtype=torch.long)

        return final_boxes, final_scores, final_labels, keep_idx
