import torch
import torch.nn as nn

from models.res50_backbone import FPNBackbone
from models.rpn_head import RPNHead
from models.roi import RoIAlignLayer
from models.bbox_head import BBoxHead
from models.reid_head import ReIDHead

class DetectorWithReID(nn.Module):
    """
    전체 Detector + ReID 파이프라인을 하나로 묶는 래퍼
    backbone → rpn → roi_align → bbox_head + reid_head
    """

    def __init__(self, backbone, rpn_head, roi_align, bbox_head, reid_head):
        super().__init__()
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.roi_align = roi_align
        self.bbox_head = bbox_head
        self.reid_head = reid_head

    @staticmethod
    def build_default(
            num_ids=500,
            in_channels=256,
            fc_dim=1024,
            num_classes=1,
            score_thresh=0.0,
            embed_dim=256,
            rpn_pre_nms=1000,
            rpn_post_nms=200,
            rpn_nms_thresh=0.7,
            strides=[8, 16, 32, 64],
            scales=[4, 8, 16],
            ratios=[0.5, 1.0, 2.0],
            device="cuda"
    ):
        """
        Build a default DetectorWithReID model instance.

        Only num_ids or key hyperparameters need to be modified when needed.
        """

        backbone = FPNBackbone()

        rpn_head = RPNHead(
            in_channels=in_channels,
            num_anchors=9,
            strides=strides,
            scales=scales,
            ratios=ratios,
            pre_nms_topk=rpn_pre_nms,
            post_nms_topk=rpn_post_nms,
            nms_thresh=rpn_nms_thresh,
        )

        roi_align = RoIAlignLayer(output_size=7)

        bbox_head = BBoxHead(
            in_channels=in_channels,
            fc_dim=fc_dim,
            num_classes=num_classes,
            score_thresh=score_thresh,
        )

        reid_head = ReIDHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_ids=num_ids,
        )

        model = DetectorWithReID(backbone, rpn_head, roi_align, bbox_head, reid_head)
        return model.to(device)

    @torch.no_grad()
    def forward(self, images):
        """
        images: (B,3,H,W)
        return: list of dicts [
            {
                'boxes': Tensor[N,4],
                'scores': Tensor[N],
                'labels': Tensor[N],
                'embeds': Tensor[N,D]
            }
        ]
        """
        self.eval()
        B, _, H, W = images.shape

        # 1) backbone → FPN
        feats = self.backbone(images)  # [P2, P3, P4, P5]

        # 2) RPN proposals
        proposals_per_img = self.rpn_head(feats, img_size=(H, W))

        outputs = []
        for b in range(B):
            proposals = proposals_per_img[b]  # proposals for image b: (N,4)

            if proposals.numel() == 0:
                outputs.append(dict(
                    boxes=torch.empty((0, 4), device=images.device),
                    scores=torch.empty((0,), device=images.device),
                    labels=torch.empty((0,), dtype=torch.long, device=images.device),
                    embeds=torch.empty((0, self.reid_head.embed_dim), device=images.device),
                ))
                continue

            # 3) RoIAlign to get fixed-size (N, C, 7, 7)
            roi_feats = self.roi_align(feats, proposals)

            # 4) BBox head → class, box
            cls_scores, bbox_preds = self.bbox_head(roi_feats)

            # 5) decode + NMS → final det boxes/scores/labels
            boxes, scores, labels, keep_idx = self.bbox_head.postprocess(
                cls_scores, bbox_preds, proposals, images[b]
            )

            # 6) ReID embedding head
            embeds_all = self.reid_head(roi_feats)
            embeds = embeds_all[keep_idx]

            outputs.append(dict(
                boxes=boxes,
                scores=scores,
                labels=labels,
                embeds=embeds
            ))
        return outputs

    def forward_train(
            self,
            images,
            gt_boxes,
            gt_ids,
            epoch: int = 0,
            global_step=None,
            warmup_iters=300,
            pos_iou_thr: float = 0.5,
            neg_iou_thr: float = 0.5,
            max_samples: int = 128
    ):
        """
        images: (B,3,H,W)
        gt_boxes:  list[Tensor(N_i,4)]  # xyxy, image 좌표
        gt_labels: list[Tensor(N_i,)]
        gt_ids:    list[Tensor(N_i,)]   # 0..num_ids-1
        epoch: 현재 epoch (train loop에서 넣어줘야 함)
        warmup_epochs: 이 epoch까지는 GT box만 사용해서 warm-up
        """

        device = images.device
        B, _, H, W = images.shape

        # 1) Backbone
        feats = self.backbone(images)  # FPN feature maps

        losses = dict()
        total_loss_cls = 0.0
        total_loss_reg = 0.0
        total_loss_reid = 0.0
        num_imgs_used = 0

        # 2) RPN으로부터 proposals 얻기 (joint 단계에서 사용)
        #  rpn_head가 단순히 proposals만 리턴한다고 가정
        # proposals_per_img = self.rpn_head(feats, img_size=(H, W))

        #  rpn_head가 loss도 지원한다면 이런 형태로 쓸 수 있음
        proposals_per_img, rpn_losses = self.rpn_head.forward_train(
            feats,
            img_size=(H, W),
            gt_boxes=gt_boxes,
        )
        total_rpn_loss = rpn_losses["loss_rpn_cls"] + rpn_losses["loss_rpn_reg"]

        for b in range(B):
            boxes_gt = gt_boxes[b].to(device)
            ids_gt = gt_ids[b].to(device)

            if boxes_gt.numel() == 0:
                continue

            # -------------------------------
            # 3) RoI 선택: warm-up vs joint
            # -------------------------------
            if global_step is None:
                use_gt_roi = True  # global_step 안 넘겨주면 항상 GT warmup
            else:
                use_gt_roi = (global_step < warmup_iters)

            if use_gt_roi:
                # warm-up: GT box를 그대로 RoI로 사용
                rois = boxes_gt
                gt_cls_targets = torch.ones(
                    rois.shape[0], dtype=torch.long, device=device
                )  # 모두 person class=1
                # GT box 그대로 유지하므로 delta=0
                gt_reg_targets = torch.zeros((rois.shape[0], 4), device=device)
                # ReID는 GT box에 대응하는 id 사용
                reid_ids = ids_gt
            else:
                # joint: RPN proposals 사용
                proposals = proposals_per_img[b].to(device)  # (M,4)
                if proposals.numel() == 0:
                    continue

                # 3-1) proposals ↔ GT box IoU 매칭
                ious = self._box_iou(proposals, boxes_gt)  # (M, G)
                max_ious, max_ids = ious.max(dim=1)  # each proposal의 best GT

                # 3-2) 양/음 샘플 구분
                pos_mask = max_ious >= pos_iou_thr
                neg_mask = max_ious < neg_iou_thr

                pos_idx = torch.nonzero(pos_mask, as_tuple=False).flatten()
                neg_idx = torch.nonzero(neg_mask, as_tuple=False).flatten()

                if pos_idx.numel() == 0:
                    # 양성 샘플이 전혀 없으면 이 이미지는 스킵
                    continue

                # 3-3) 샘플링 (양/음 합해서 max_samples로 제한)
                num_pos = min(pos_idx.numel(), max_samples // 2)
                num_neg = min(neg_idx.numel(), max_samples - num_pos)

                perm_pos = torch.randperm(pos_idx.numel(), device=device)[:num_pos]
                perm_neg = torch.randperm(neg_idx.numel(), device=device)[:num_neg]

                sel_pos = pos_idx[perm_pos]
                sel_neg = neg_idx[perm_neg]

                sel_idx = torch.cat([sel_pos, sel_neg], dim=0)
                rois = proposals[sel_idx]  # (N,4)

                # 3-4) cls target: pos → 1(person), neg → 0(background)
                gt_cls_targets = torch.zeros(rois.shape[0], dtype=torch.long, device=device)
                gt_cls_targets[:num_pos] = 1

                # 3-5) reg target: positive에 대해서만 GT box를 기준으로 delta 인코딩
                matched_gt = boxes_gt[max_ids[sel_pos]]  # (num_pos,4)
                base_proposals = proposals[sel_pos]  # (num_pos,4)

                reg_targets_pos = self._encode_boxes(base_proposals, matched_gt)  # (num_pos,4)

                gt_reg_targets = torch.zeros((rois.shape[0], 4), device=device)
                gt_reg_targets[:num_pos] = reg_targets_pos

                # 3-6) ReID target: positive RoI에 대해서만 GT id 사용
                reid_ids = ids_gt[max_ids[sel_pos]]  # (num_pos,)

            # -------------------------------------------------
            # 4) RoIAlign → BBoxHead / ReIDHead → 손실 계산
            # -------------------------------------------------
            roi_feats = self.roi_align(feats, rois)  # (N,256,7,7)

            # BBoxHead
            cls_scores, bbox_preds = self.bbox_head(roi_feats)
            loss_det = self.bbox_head.loss(
                cls_scores, bbox_preds, gt_cls_targets, gt_reg_targets
            )
            loss_cls = loss_det["loss_cls"]
            loss_reg = loss_det["loss_reg"]

            # ReIDHead (positive RoI만 사용)
            embeds = self.reid_head(roi_feats)  # (N, D)

            if use_gt_roi:
                # warm-up때는 모든 RoI가 GT라서 모두 positive
                embeds_pos = embeds
                ids_pos = reid_ids
            else:
                # joint 단계에서는 앞쪽 num_pos가 positive
                if num_pos > 0:
                    embeds_pos = embeds[:num_pos]
                    ids_pos = reid_ids
                else:
                    continue

            loss_reid_dict = self.reid_head.loss(embeds_pos, ids_pos)
            loss_reid = loss_reid_dict["loss_reid"]

            total_loss_cls += loss_cls
            total_loss_reg += loss_reg
            total_loss_reid += loss_reid
            num_imgs_used += 1

        if num_imgs_used == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "loss_rpn": zero,
                "loss_cls": zero,
                "loss_reg": zero,
                "loss_reid": zero,
                "loss_total": zero
            }

        # 이미지 수로 평균
        loss_cls_mean = total_loss_cls / num_imgs_used
        loss_reg_mean = total_loss_reg / num_imgs_used
        loss_reid_mean = total_loss_reid / num_imgs_used

        # RPN loss를 쓴다면 여기에 포함
        loss_rpn_mean = total_rpn_loss / max(num_imgs_used, 1) if total_rpn_loss != 0 else 0.0

        # 최종 loss 조합 (가중치는 취향껏 조정)
        loss_total = (
                loss_cls_mean * 1.0 +
                loss_reg_mean * 1.0 +
                loss_reid_mean * 0.25 +
                (loss_rpn_mean if isinstance(loss_rpn_mean, torch.Tensor) else 0.0)
        )

        return dict(
            loss_rpn=loss_rpn_mean if isinstance(loss_rpn_mean, torch.Tensor) else torch.tensor(0.0, device=device),
            loss_cls=loss_cls_mean,
            loss_reg=loss_reg_mean,
            loss_reid=loss_reid_mean,
            loss_total=loss_total
        )

    @staticmethod
    def _box_iou(boxes1, boxes2):
        """
        boxes1: (M,4), boxes2: (G,4)  [x1,y1,x2,y2]
        return: IoU matrix (M,G)
        """
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
                (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
                (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (M,G,2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (M,G,2)

        wh = (rb - lt).clamp(min=0)  # (M,G,2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # (M,G)

        union = area1[:, None] + area2 - inter
        iou = inter / union.clamp(min=1e-6)
        return iou

    @staticmethod
    def _encode_boxes(proposals, gt_boxes):
        """
        proposals: (N,4), gt_boxes: (N,4)
        Faster R-CNN 스타일 delta encoding
        """
        # proposals
        pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
        ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
        px = proposals[:, 0] + 0.5 * pw
        py = proposals[:, 1] + 0.5 * ph

        # gt
        gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
        gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)
        gx = gt_boxes[:, 0] + 0.5 * gw
        gy = gt_boxes[:, 1] + 0.5 * gh

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)

        deltas = torch.stack([dx, dy, dw, dh], dim=1)
        return deltas