import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from .anchor import AnchorGenerator

class RPNHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 num_anchors=3,
                 strides=[8, 16, 32, 64],
                 scales=[4, 8, 16],      # stride와 곱해져서 실제 anchor 크기 결정
                 ratios=[0.5, 1.0, 2.0],
                 pre_nms_topk=1000,
                 post_nms_topk=300,
                 nms_thresh=0.7,
                 min_box_size=5.0,
                 score_thresh=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred  = nn.Conv2d(in_channels, num_anchors * 4, 1)

        self.anchor_generator = AnchorGenerator(
            strides=strides,
            scales=scales,
            ratios=ratios
        )

        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_box_size = min_box_size
        self.score_thresh = score_thresh

    def _flatten_predictions(self, cls_logits_list, bbox_pred_list):
        """
        FPN level별 cls/reg 출력을 flatten해서
        이미지 단위 텐서로 합치는 함수.
        """
        batch_size = cls_logits_list[0].shape[0]
        all_objectness = []
        all_bbox_deltas = []

        for cls_logits, bbox_pred in zip(cls_logits_list, bbox_pred_list):
            # cls_logits: (B, A, H, W)
            B, A, H, W = cls_logits.shape
            # (B, H, W, A) -> (B, H*W*A)
            cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(B, -1)
            objectness = cls_logits.sigmoid()

            # bbox_pred: (B, 4A, H, W)
            bbox_pred = bbox_pred.view(B, A, 4, H, W)
            bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2).reshape(B, -1, 4)  # (B, H*W*A, 4)

            all_objectness.append(objectness)
            all_bbox_deltas.append(bbox_pred)

        # concat over all levels
        all_objectness = torch.cat(all_objectness, dim=1)    # (B, N)
        all_bbox_deltas = torch.cat(all_bbox_deltas, dim=1)  # (B, N, 4)

        return all_objectness, all_bbox_deltas

    def _decode_boxes(self, anchors, deltas):
        """
        anchors: (N,4)  xyxy
        deltas:  (N,4)  [dx, dy, dw, dh]
        """
        # anchors → cx,cy,w,h
        widths  = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x   = anchors[:, 0] + 0.5 * widths
        ctr_y   = anchors[:, 1] + 0.5 * heights

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
        return decoded

    def _clip_boxes(self, boxes, img_h, img_w):
        boxes[:, 0].clamp_(min=0, max=img_w - 1)
        boxes[:, 1].clamp_(min=0, max=img_h - 1)
        boxes[:, 2].clamp_(min=0, max=img_w - 1)
        boxes[:, 3].clamp_(min=0, max=img_h - 1)
        return boxes

    def _remove_small_boxes(self, boxes):
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        keep = (w >= self.min_box_size) & (h >= self.min_box_size)
        return keep

    @torch.no_grad()
    def forward(self, feats, img_size):
        """
        feats: FPN feature list [P2,P3,...], each (B,C,H,W)
        img_size: (H,W) 원본 이미지 크기
        return: List[Tensor] proposals_per_image, length=B
        """
        B = feats[0].shape[0]
        device = feats[0].device

        # 1) 각 level별 head forward
        cls_logits_list = []
        bbox_pred_list  = []

        for f in feats:
            t = F.relu(self.conv(f))
            cls_logits = self.cls_logits(t)  # (B,A,H,W)
            bbox_pred  = self.bbox_pred(t)   # (B,4A,H,W)
            cls_logits_list.append(cls_logits)
            bbox_pred_list.append(bbox_pred)

        # 2) Anchor 생성 (이미지 기준, 배치 공유)
        feat_shapes = [(f.shape[2], f.shape[3]) for f in feats]  # (H_i,W_i)
        anchors_per_level = self.anchor_generator.grid_anchors(feat_shapes)
        anchors = torch.cat(anchors_per_level, dim=0).to(device)  # (N,4)

        # 3) flatten predictions
        all_objectness, all_bbox_deltas = self._flatten_predictions(
            cls_logits_list, bbox_pred_list
        )
        # all_objectness: (B,N), all_bbox_deltas: (B,N,4)

        proposals_per_image = []
        img_h, img_w = img_size

        for b in range(B):
            scores = all_objectness[b]       # (N,)
            deltas = all_bbox_deltas[b]      # (N,4)

            # score threshold
            keep = scores > self.score_thresh
            scores = scores[keep]
            deltas = deltas[keep]
            anchors_b = anchors[keep]

            if scores.numel() == 0:
                proposals_per_image.append(torch.empty((0,4), device=device))
                continue

            # pre_nms_topk
            num_topk = min(self.pre_nms_topk, scores.size(0))
            topk_vals, topk_idx = scores.topk(num_topk, sorted=True)
            scores = topk_vals
            deltas = deltas[topk_idx]
            anchors_b = anchors_b[topk_idx]

            # decode & clip
            boxes = self._decode_boxes(anchors_b, deltas)
            boxes = self._clip_boxes(boxes, img_h, img_w)

            # remove small boxes
            keep2 = self._remove_small_boxes(boxes)
            boxes  = boxes[keep2]
            scores = scores[keep2]

            if boxes.numel() == 0:
                proposals_per_image.append(torch.empty((0,4), device=device))
                continue

            # NMS
            keep_idx = nms(boxes, scores, self.nms_thresh)
            keep_idx = keep_idx[: self.post_nms_topk]

            proposals = boxes[keep_idx]
            proposals_per_image.append(proposals)

        return proposals_per_image

    def forward_train(
            self,
            feats,
            img_size,
            gt_boxes,
            pos_iou_thr: float = 0.5,
            neg_iou_thr: float = 0.5,
            batch_size_per_img: int = 256,
            positive_fraction: float = 0.5,
    ):
        """
        feats: FPN feature list [P2, P3, ...], each (B,C,H,W)
        img_size: (H, W) 원본 이미지 크기
        gt_boxes: list[Tensor(N_i,4)]  각 이미지별 GT box (xyxy)

        return:
          proposals_per_image: List[Tensor(N_i,4)]
          rpn_losses: dict(loss_rpn_cls, loss_rpn_reg)
        """
        B = feats[0].shape[0]
        device = feats[0].device
        img_h, img_w = img_size

        # 1) 각 level별 head forward (no no_grad!)
        cls_logits_list = []
        bbox_pred_list = []
        feat_shapes = []

        for f in feats:
            t = F.relu(self.conv(f))
            cls_logits = self.cls_logits(t)  # (B,A,H,W)
            bbox_pred = self.bbox_pred(t)  # (B,4A,H,W)
            cls_logits_list.append(cls_logits)
            bbox_pred_list.append(bbox_pred)
            feat_shapes.append((f.shape[2], f.shape[3]))

        # 2) Anchor 생성 (이미지 기준, 배치 공유)
        anchors_per_level = self.anchor_generator.grid_anchors(feat_shapes)
        anchors = torch.cat(anchors_per_level, dim=0).to(device)  # (N,4) xyxy 전체 anchor

        # 3) 이미지별 RPN loss 계산 + proposals 생성
        loss_rpn_cls = 0.0
        loss_rpn_reg = 0.0
        num_images_used = 0
        proposals_per_image = []

        for b in range(B):
            # 3-1) 이 이미지에 대한 cls_logits, bbox_pred flatten
            obj_list = []  # objectness logits
            deltas_list = []

            for cls_logits, bbox_pred in zip(cls_logits_list, bbox_pred_list):
                # cls_logits: (B,A,H,W) → (H*W*A,)
                cls_b = cls_logits[b:b + 1]  # (1,A,H,W)
                _, A, Hf, Wf = cls_b.shape
                cls_b = cls_b.permute(0, 2, 3, 1).reshape(-1)  # (Hf*Wf*A,)

                # bbox_pred: (B,4A,H,W) → (H*W*A,4)
                bbox_b = bbox_pred[b:b + 1]  # (1,4A,H,W)
                bbox_b = bbox_b.view(1, A, 4, Hf, Wf)
                bbox_b = bbox_b.permute(0, 3, 4, 1, 2).reshape(-1, 4)  # (Hf*Wf*A,4)

                obj_list.append(cls_b)
                deltas_list.append(bbox_b)

            objectness = torch.cat(obj_list, dim=0)  # (N_total,)
            pred_deltas = torch.cat(deltas_list, dim=0)  # (N_total,4)
            num_anchors = anchors.shape[0]
            assert objectness.shape[0] == num_anchors
            assert pred_deltas.shape[0] == num_anchors

            gt = gt_boxes[b].to(device)
            # ----------------------------------
            # 3-2) anchor ↔ GT IoU 기반 라벨링
            # ----------------------------------
            if gt.numel() == 0:
                # GT가 없으면, 모든 anchor를 negative로 간주
                labels = torch.zeros(num_anchors, dtype=torch.float32, device=device)
                bbox_targets = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
            else:
                ious = self._box_iou(anchors, gt)  # (N_total, G)
                max_ious, max_ids = ious.max(dim=1)  # anchor별 best GT

                # -1: ignore, 0: negative, 1: positive
                labels = torch.full((num_anchors,), -1, dtype=torch.float32, device=device)

                # negative: IoU < neg_thr
                labels[max_ious < neg_iou_thr] = 0.0

                # positive: IoU >= pos_thr
                labels[max_ious >= pos_iou_thr] = 1.0

                # (선택) 각각의 GT는 최소한 하나의 positive anchor를 갖도록 보장해도 됨
                # for g in range(gt.shape[0]):
                #     gt_ious = ious[:, g]
                #     best_anchor = torch.argmax(gt_ious)
                #     labels[best_anchor] = 1.0
                #     max_ids[best_anchor] = g

                # bbox regression target: positive anchor에 대해 GT box를 encode
                bbox_targets = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
                pos_inds_all = torch.nonzero(labels == 1.0, as_tuple=False).flatten()
                if pos_inds_all.numel() > 0:
                    matched_gt = gt[max_ids[pos_inds_all]]  # (N_pos,4)
                    base_anchors = anchors[pos_inds_all]  # (N_pos,4)
                    bbox_targets_pos = self._encode_boxes(base_anchors, matched_gt)
                    bbox_targets[pos_inds_all] = bbox_targets_pos

            # ----------------------------------
            # 3-3) 샘플링 (pos/neg에서만 loss 계산)
            # ----------------------------------
            pos_inds = torch.nonzero(labels == 1.0, as_tuple=False).flatten()
            neg_inds = torch.nonzero(labels == 0.0, as_tuple=False).flatten()

            num_pos = int(batch_size_per_img * positive_fraction)
            num_pos = min(num_pos, pos_inds.numel())
            num_neg = batch_size_per_img - num_pos
            num_neg = min(num_neg, neg_inds.numel())

            if num_pos == 0 and num_neg == 0:
                # 이 이미지는 스킵
                proposals = self._proposals_from_anchors(
                    anchors, objectness, pred_deltas, img_h, img_w
                )
                proposals_per_image.append(proposals)
                continue

            if num_pos > 0:
                perm_pos = torch.randperm(pos_inds.numel(), device=device)[:num_pos]
                pos_inds = pos_inds[perm_pos]
            if num_neg > 0:
                perm_neg = torch.randperm(neg_inds.numel(), device=device)[:num_neg]
                neg_inds = neg_inds[perm_neg]

            sampled_inds = torch.cat([pos_inds, neg_inds], dim=0)
            sampled_labels = torch.zeros_like(sampled_inds, dtype=torch.float32, device=device)
            sampled_labels[:num_pos] = 1.0

            # ----------------------------------
            # 3-4) classification loss (BCE with logits)
            # ----------------------------------
            objectness_sampled = objectness[sampled_inds]  # (N_sample,)
            loss_cls_img = F.binary_cross_entropy_with_logits(
                objectness_sampled, sampled_labels, reduction="mean"
            )

            # ----------------------------------
            # 3-5) regression loss (Smooth L1, pos만)
            # ----------------------------------
            if num_pos > 0:
                pred_deltas_pos = pred_deltas[pos_inds]  # (N_pos,4)
                bbox_targets_pos = bbox_targets[pos_inds]  # (N_pos,4)
                loss_reg_img = F.smooth_l1_loss(
                    pred_deltas_pos, bbox_targets_pos, reduction="mean"
                )
            else:
                loss_reg_img = torch.tensor(0.0, device=device)

            loss_rpn_cls += loss_cls_img
            loss_rpn_reg += loss_reg_img
            num_images_used += 1

            # ----------------------------------
            # 3-6) 이 이미지에 대한 proposals 생성 (NMS)
            # ----------------------------------
            proposals = self._proposals_from_anchors(
                anchors, objectness.detach().sigmoid(), pred_deltas.detach(), img_h, img_w
            )
            proposals_per_image.append(proposals)

        if num_images_used == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return proposals_per_image, dict(
                loss_rpn_cls=zero, loss_rpn_reg=zero
            )

        loss_rpn_cls = loss_rpn_cls / num_images_used
        loss_rpn_reg = loss_rpn_reg / num_images_used

        rpn_losses = dict(
            loss_rpn_cls=loss_rpn_cls,
            loss_rpn_reg=loss_rpn_reg
        )
        return proposals_per_image, rpn_losses

    def _proposals_from_anchors(self, anchors, scores, deltas, img_h, img_w):
        """
        anchors: (N,4)
        scores: (N,)   (이미 sigmoid 통과한 점수라고 가정)
        deltas: (N,4)
        """
        device = anchors.device

        # score threshold
        keep = scores > self.score_thresh
        if keep.sum() == 0:
            return torch.empty((0, 4), device=device)

        scores_k = scores[keep]
        anchors_k = anchors[keep]
        deltas_k = deltas[keep]

        # pre_nms_topk
        num_topk = min(self.pre_nms_topk, scores_k.size(0))
        topk_vals, topk_idx = scores_k.topk(num_topk, sorted=True)
        scores_k = topk_vals
        anchors_k = anchors_k[topk_idx]
        deltas_k = deltas_k[topk_idx]

        # decode & clip
        boxes = self._decode_boxes(anchors_k, deltas_k)
        boxes = self._clip_boxes(boxes, img_h, img_w)

        # remove small boxes
        keep2 = self._remove_small_boxes(boxes)
        boxes = boxes[keep2]
        scores_k = scores_k[keep2]

        if boxes.numel() == 0:
            return torch.empty((0, 4), device=device)

        # NMS
        keep_idx = nms(boxes, scores_k, self.nms_thresh)
        keep_idx = keep_idx[: self.post_nms_topk]

        proposals = boxes[keep_idx]
        return proposals

    @staticmethod
    def _box_iou(boxes1, boxes2):
        """
        boxes1: (N,4), boxes2: (M,4), xyxy
        """
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
                (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
                (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)

        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]

        union = area1[:, None] + area2 - inter
        iou = inter / union.clamp(min=1e-6)
        return iou

    @staticmethod
    def _encode_boxes(anchors, gt_boxes):
        """
        anchors: (N,4), gt_boxes: (N,4)  [x1,y1,x2,y2]
        Faster R-CNN 스타일 delta encoding
        """
        pw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
        ph = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
        px = anchors[:, 0] + 0.5 * pw
        py = anchors[:, 1] + 0.5 * ph

        gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
        gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)
        gx = gt_boxes[:, 0] + 0.5 * gw
        gy = gt_boxes[:, 1] + 0.5 * gh

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)

        return torch.stack([dx, dy, dw, dh], dim=1)