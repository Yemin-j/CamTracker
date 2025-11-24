import torch
import torch.nn as nn

from models.res50_backbone import FPNBackbone
from models.rpn_head import RPNHead
from models.roi import RoIAlignLayer
from models.bbox_head import BBoxHead
from models.reid_head import ReIDHead


class QDTrackDetector(nn.Module):
    """
    QDTrack: Quasi-Dense Similarity Learning for Multiple Object Tracking

    Key differences from standard detector:
    1. Temporal pair processing (key frame + reference frame)
    2. Quasi-dense matching between proposals across frames
    3. Multi-positive contrastive learning for ReID
    4. Pure joint learning (no staged warm-up)

    Architecture: backbone → rpn → roi_align → bbox_head + reid_head
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
            in_channels=256,
            fc_dim=1024,
            num_classes=1,
            score_thresh=0.05,
            nms_thresh=0.5,
            max_dets=100,
            embed_dim=256,
            temperature=0.1,
            rpn_pre_nms=1000,
            rpn_post_nms=200,
            rpn_nms_thresh=0.7,
            strides=[8, 16, 32, 64],
            scales=[4, 8, 16],
            ratios=[0.5, 1.0, 2.0],
            pretrained=True,
            device="cuda"
    ):
        """
        Build a default QDTrack model instance.

        Args:
            in_channels: FPN output channels (256)
            fc_dim: Hidden dimension for detection head (1024)
            num_classes: Number of foreground classes (1 for person-only)
            score_thresh: Score threshold for filtering detections
            nms_thresh: NMS IoU threshold
            max_dets: Maximum detections per image
            embed_dim: ReID embedding dimension (256)
            temperature: Temperature for contrastive learning (0.07)
            rpn_pre_nms: Top-K proposals before NMS (1000)
            rpn_post_nms: Top-K proposals after NMS (200)
            rpn_nms_thresh: RPN NMS threshold (0.7)
            strides: FPN strides [8, 16, 32, 64] (actually [4,8,16,32] for P2-P5)
            scales: Anchor scales [4, 8, 16]
            ratios: Anchor aspect ratios [0.5, 1.0, 2.0]
            pretrained: Use pretrained ResNet-50 backbone
            device: Device to place model on

        Note: num_ids parameter removed - QDTrack uses contrastive learning,
              not ID classification.
        """

        # Backbone: ResNet-50 + FPN
        backbone = FPNBackbone(pretrained=pretrained)

        # RPN Head with QDTrack parameters
        rpn_head = RPNHead(
            in_channels=in_channels,
            num_anchors=9,  # 3 scales × 3 ratios
            strides=[4, 8, 16, 32],  # Actual strides for P2, P3, P4, P5
            scales=scales,
            ratios=ratios,
            pre_nms_topk=rpn_pre_nms,
            post_nms_topk=rpn_post_nms,
            nms_thresh=rpn_nms_thresh,
        )

        # RoI Align Layer (FPN-based multi-scale pooling)
        roi_align = RoIAlignLayer(output_size=7)

        # Detection Head
        bbox_head = BBoxHead(
            in_channels=in_channels,
            fc_dim=fc_dim,
            num_classes=num_classes,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            max_dets=max_dets,
        )

        # ReID Head (Multi-positive contrastive learning)
        reid_head = ReIDHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            temperature=temperature,
        )

        model = QDTrackDetector(backbone, rpn_head, roi_align, bbox_head, reid_head)
        return model.to(device)

    @torch.no_grad()
    def forward(self, images):
        """
        Inference forward pass (single frame)

        Args:
            images: (B,3,H,W) input images

        Returns:
            list of dicts [
                {
                    'boxes': Tensor[N,4] in xyxy format,
                    'scores': Tensor[N] confidence scores,
                    'labels': Tensor[N] class labels (all 1 for person),
                    'embeds': Tensor[N,D] ReID embeddings
                }
            ]
        """
        self.eval()
        B, _, H, W = images.shape
        device = images.device

        # 1) Backbone → FPN
        feats = self.backbone(images)  # [P2, P3, P4, P5]

        # 2) RPN proposals
        proposals_per_img = self.rpn_head(feats, img_size=(H, W))

        outputs = []
        for b in range(B):
            proposals = proposals_per_img[b]  # (N,4) proposals for image b

            if proposals.numel() == 0:
                # No proposals - return empty results
                outputs.append(dict(
                    boxes=torch.empty((0, 4), device=device, dtype=torch.float32),
                    scores=torch.empty((0,), device=device, dtype=torch.float32),
                    labels=torch.empty((0,), dtype=torch.long, device=device),
                    embeds=torch.empty((0, self.reid_head.embed_dim), device=device, dtype=torch.float32),
                ))
                continue

            # 3) RoIAlign to get fixed-size features (N, C, 7, 7)
            roi_feats = self.roi_align(feats, proposals)

            # Safety check: roi_feats should match proposals
            if roi_feats.shape[0] != proposals.shape[0]:
                raise RuntimeError(
                    f"RoI feature count mismatch: {roi_feats.shape[0]} vs {proposals.shape[0]}"
                )

            # 4) BBox head → classification and regression
            cls_scores, bbox_preds = self.bbox_head(roi_feats)

            # 5) Decode + NMS → final detections
            boxes, scores, labels, keep_idx = self.bbox_head.postprocess(
                cls_scores, bbox_preds, proposals, images[b]
            )

            # 6) ReID embedding head
            if keep_idx.numel() > 0:
                embeds_all = self.reid_head(roi_feats)
                embeds = embeds_all[keep_idx]
            else:
                embeds = torch.empty((0, self.reid_head.embed_dim), device=device, dtype=torch.float32)

            outputs.append(dict(
                boxes=boxes,
                scores=scores,
                labels=labels,
                embeds=embeds
            ))
        return outputs

    def forward_train(
            self,
            images_key,
            images_ref,
            gt_boxes_key,
            gt_boxes_ref,
            gt_ids_key,
            gt_ids_ref,
            pos_iou_thr: float = 0.7,  # QDTrack uses 0.7
            neg_iou_thr: float = 0.3,  # QDTrack uses 0.3
            num_samples_key: int = 128,  # V: training samples from key frame
            num_samples_ref: int = 256,  # K: contrastive targets from ref frame
    ):
        """
        QDTrack training forward pass with temporal pair processing

        Args:
            images_key: (B,3,H,W) Key frame
            images_ref: (B,3,H,W) Reference frame (temporal neighbor, k ∈ [-3,3])
            gt_boxes_key: list[Tensor(N_i,4)] GT boxes for key frame (xyxy)
            gt_boxes_ref: list[Tensor(M_i,4)] GT boxes for reference frame (xyxy)
            gt_ids_key: list[Tensor(N_i,)] GT track IDs for key frame
            gt_ids_ref: list[Tensor(M_i,)] GT track IDs for reference frame
            pos_iou_thr: IoU threshold for positive matches (0.7 in paper)
            neg_iou_thr: IoU threshold for negative matches (0.3 in paper)
            num_samples_key: Number of RoIs to sample from key frame (V=128)
            num_samples_ref: Number of RoIs to sample from ref frame (K=256)

        Returns:
            dict with losses:
                - loss_rpn: RPN loss (cls + reg)
                - loss_cls: Detection classification loss
                - loss_reg: Detection regression loss
                - loss_contrastive: Multi-positive contrastive loss
                - loss_aux: Auxiliary L2 loss
                - loss_reid: Total ReID loss (contrastive + aux)
                - loss_total: Total weighted loss
        """

        device = images_key.device
        B, _, H, W = images_key.shape

        # Validate inputs
        assert images_ref.shape == images_key.shape, \
            f"Image shape mismatch: key {images_key.shape} vs ref {images_ref.shape}"
        assert len(gt_boxes_key) == B and len(gt_boxes_ref) == B, \
            f"Batch size mismatch in GT boxes"
        assert len(gt_ids_key) == B and len(gt_ids_ref) == B, \
            f"Batch size mismatch in GT IDs"

        # ========================================
        # 1) KEY FRAME FORWARD
        # ========================================
        feats_key = self.backbone(images_key)
        self.rpn_head.debug_frames = images_key
        proposals_key, rpn_losses_key = self.rpn_head.forward_train(
            feats_key,
            img_size=(H, W),
            gt_boxes=gt_boxes_key,
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr,
        )

        # ========================================
        # 2) REFERENCE FRAME FORWARD
        # ========================================
        feats_ref = self.backbone(images_ref)
        proposals_ref, rpn_losses_ref = self.rpn_head.forward_train(
            feats_ref,
            img_size=(H, W),
            gt_boxes=gt_boxes_ref,
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr,
        )

        # ========================================
        # 3) ACCUMULATE LOSSES
        # ========================================
        total_loss_cls = 0.0
        total_loss_reg = 0.0
        total_loss_contrastive = 0.0
        total_loss_aux = 0.0
        num_imgs_used = 0

        # ========================================
        # 4) BATCH LOOP: QUASI-DENSE MATCHING
        # ========================================
        for b in range(B):
            gt_boxes_k = gt_boxes_key[b].to(device)
            gt_boxes_r = gt_boxes_ref[b].to(device)
            gt_ids_k = gt_ids_key[b].to(device)
            gt_ids_r = gt_ids_ref[b].to(device)

            # Skip if no GT boxes in either frame
            if gt_boxes_k.numel() == 0 or gt_boxes_r.numel() == 0:
                continue

            proposals_k = proposals_key[b].to(device)  # (M_k, 4)
            proposals_r = proposals_ref[b].to(device)  # (M_r, 4)

            # Skip if no proposals in either frame
            if proposals_k.numel() == 0 or proposals_r.numel() == 0:
                continue

            # ----------------------------------------
            # 4-1) Sample RoIs from key frame (V samples)
            # ----------------------------------------
            ious_k = self._box_iou(proposals_k, gt_boxes_k)  # (M_k, G_k)
            max_ious_k, max_ids_k = ious_k.max(dim=1)

            pos_mask_k = max_ious_k >= pos_iou_thr
            neg_mask_k = max_ious_k < neg_iou_thr

            pos_idx_k = torch.nonzero(pos_mask_k, as_tuple=False).flatten()
            neg_idx_k = torch.nonzero(neg_mask_k, as_tuple=False).flatten()

            # Must have at least one positive
            if pos_idx_k.numel() == 0:
                continue

            # IoU-balanced sampling (pos:neg = 1:1)
            num_pos_k = min(pos_idx_k.numel(), num_samples_key // 2)
            num_neg_k = min(neg_idx_k.numel(), num_samples_key - num_pos_k)

            if num_pos_k > 0:
                perm_pos_k = torch.randperm(pos_idx_k.numel(), device=device)[:num_pos_k]
                sel_pos_k = pos_idx_k[perm_pos_k]
            else:
                sel_pos_k = torch.empty(0, dtype=torch.long, device=device)

            if num_neg_k > 0:
                perm_neg_k = torch.randperm(neg_idx_k.numel(), device=device)[:num_neg_k]
                sel_neg_k = neg_idx_k[perm_neg_k]
            else:
                sel_neg_k = torch.empty(0, dtype=torch.long, device=device)

            sel_idx_k = torch.cat([sel_pos_k, sel_neg_k], dim=0)
            rois_k = proposals_k[sel_idx_k]  # (V, 4)
            V = rois_k.shape[0]

            # ----------------------------------------
            # 4-2) Sample RoIs from reference frame (K samples)
            # ----------------------------------------
            ious_r = self._box_iou(proposals_r, gt_boxes_r)  # (M_r, G_r)
            max_ious_r, max_ids_r = ious_r.max(dim=1)

            pos_mask_r = max_ious_r >= pos_iou_thr
            neg_mask_r = max_ious_r < neg_iou_thr

            pos_idx_r = torch.nonzero(pos_mask_r, as_tuple=False).flatten()
            neg_idx_r = torch.nonzero(neg_mask_r, as_tuple=False).flatten()

            # Need at least some positive samples in reference
            if pos_idx_r.numel() == 0:
                continue

            num_pos_r = min(pos_idx_r.numel(), num_samples_ref // 2)
            num_neg_r = min(neg_idx_r.numel(), num_samples_ref - num_pos_r)

            if num_pos_r > 0:
                perm_pos_r = torch.randperm(pos_idx_r.numel(), device=device)[:num_pos_r]
                sel_pos_r = pos_idx_r[perm_pos_r]
            else:
                sel_pos_r = torch.empty(0, dtype=torch.long, device=device)

            if num_neg_r > 0:
                perm_neg_r = torch.randperm(neg_idx_r.numel(), device=device)[:num_neg_r]
                sel_neg_r = neg_idx_r[perm_neg_r]
            else:
                sel_neg_r = torch.empty(0, dtype=torch.long, device=device)

            sel_idx_r = torch.cat([sel_pos_r, sel_neg_r], dim=0)
            rois_r = proposals_r[sel_idx_r]  # (K, 4)
            K = rois_r.shape[0]

            # ----------------------------------------
            # 4-3) Quasi-Dense Matching Matrix (V, K)
            # ----------------------------------------
            match_matrix = torch.zeros((V, K), device=device, dtype=torch.float32)

            if num_pos_k > 0 and num_pos_r > 0:
                # Get GT IDs for positive RoIs
                matched_ids_k = gt_ids_k[max_ids_k[sel_pos_k]]  # (num_pos_k,)
                matched_ids_r = gt_ids_r[max_ids_r[sel_pos_r]]  # (num_pos_r,)

                # Vectorized ID comparison: (num_pos_k, 1) == (1, num_pos_r)
                id_match = (matched_ids_k.unsqueeze(1) == matched_ids_r.unsqueeze(0)).float()

                # Place into match_matrix
                match_matrix[:num_pos_k, :num_pos_r] = id_match

            # ----------------------------------------
            # 4-4) Detection Loss (BBox Head) - Key Frame Only
            # ----------------------------------------
            roi_feats_k = self.roi_align(feats_key, rois_k)  # (V, C, 7, 7)

            # Safety check
            if roi_feats_k.shape[0] != V:
                raise RuntimeError(
                    f"RoI feature mismatch in key frame: {roi_feats_k.shape[0]} != {V}"
                )

            cls_scores_k, bbox_preds_k = self.bbox_head(roi_feats_k)

            # Classification targets
            gt_cls_targets_k = torch.zeros(V, dtype=torch.long, device=device)
            gt_cls_targets_k[:num_pos_k] = 1  # positive → class 1 (person)

            # Regression targets (only for positives)
            gt_reg_targets_k = torch.zeros((V, 4), device=device, dtype=torch.float32)
            if num_pos_k > 0:
                matched_gt_k = gt_boxes_k[max_ids_k[sel_pos_k]]
                base_proposals_k = proposals_k[sel_pos_k]
                reg_targets_pos_k = self._encode_boxes(base_proposals_k, matched_gt_k)
                gt_reg_targets_k[:num_pos_k] = reg_targets_pos_k

            # Compute detection loss
            loss_det_k = self.bbox_head.loss(
                cls_scores_k, bbox_preds_k, gt_cls_targets_k, gt_reg_targets_k
            )
            loss_cls = loss_det_k["loss_cls"]
            loss_reg = loss_det_k["loss_reg"]

            # ----------------------------------------
            # 4-5) ReID Loss (Contrastive Learning)
            # ----------------------------------------
            embeds_k = self.reid_head(roi_feats_k)  # (V, D)

            roi_feats_r = self.roi_align(feats_ref, rois_r)  # (K, C, 7, 7)

            # Safety check
            if roi_feats_r.shape[0] != K:
                raise RuntimeError(
                    f"RoI feature mismatch in ref frame: {roi_feats_r.shape[0]} != {K}"
                )

            embeds_r = self.reid_head(roi_feats_r)  # (K, D)

            # Multi-positive contrastive loss
            loss_reid_dict = self.reid_head.loss(embeds_k, embeds_r, match_matrix)
            loss_contrastive = loss_reid_dict["loss_contrastive"]
            loss_aux = loss_reid_dict["loss_aux"]

            # ----------------------------------------
            # 4-6) Accumulate
            # ----------------------------------------
            total_loss_cls += loss_cls
            total_loss_reg += loss_reg
            total_loss_contrastive += loss_contrastive
            total_loss_aux += loss_aux
            num_imgs_used += 1

        # ========================================
        # 5) AVERAGE AND COMBINE LOSSES
        # ========================================

        # RPN losses (always computed)
        loss_rpn_mean = (
                                rpn_losses_key["loss_rpn_cls"] + rpn_losses_key["loss_rpn_reg"] +
                                rpn_losses_ref["loss_rpn_cls"] + rpn_losses_ref["loss_rpn_reg"]
                        ) / 2.0

        if num_imgs_used == 0:
            # No valid samples - return minimal losses with gradient
            zero = torch.tensor(0.0, device=device, requires_grad=True)

            # Ensure gradient flow by touching model parameters
            dummy_loss = sum(p.sum() for p in self.parameters()) * 0.0

            return {
                "loss_rpn": loss_rpn_mean + dummy_loss,
                "loss_cls": zero + dummy_loss,
                "loss_reg": zero + dummy_loss,
                "loss_contrastive": zero + dummy_loss,
                "loss_aux": zero + dummy_loss,
                "loss_reid": zero + dummy_loss,
                "loss_total": loss_rpn_mean + dummy_loss
            }

        # Average detection and ReID losses
        loss_cls_mean = total_loss_cls / num_imgs_used
        loss_reg_mean = total_loss_reg / num_imgs_used
        loss_contrastive_mean = total_loss_contrastive / num_imgs_used
        loss_aux_mean = total_loss_aux / num_imgs_used

        # Total ReID loss
        loss_reid_mean = loss_contrastive_mean + loss_aux_mean

        # QDTrack loss weights (Eq. 7 in paper)
        # L_total = L_det + γ1 * L_cont + γ2 * L_aux
        # Paper uses γ1=0.25, γ2=1.0
        loss_total = (
                loss_rpn_mean * 1.0 +  # RPN loss
                loss_cls_mean * 1.0 +  # Detection classification
                loss_reg_mean * 1.0 +  # Detection regression
                loss_contrastive_mean * 0.2 +  # Contrastive loss (γ1=0.25)
                loss_aux_mean * 1.0  # Auxiliary L2 loss (γ2=1.0)
        )

        return dict(
            loss_rpn=loss_rpn_mean,
            loss_cls=loss_cls_mean,
            loss_reg=loss_reg_mean,
            loss_contrastive=loss_contrastive_mean,
            loss_aux=loss_aux_mean,
            loss_reid=loss_reid_mean,
            loss_total=loss_total
        )

    @staticmethod
    def _box_iou(boxes1, boxes2):
        """
        Compute IoU matrix between two sets of boxes

        Args:
            boxes1: (M,4) [x1,y1,x2,y2]
            boxes2: (G,4) [x1,y1,x2,y2]

        Returns:
            IoU matrix (M,G)
        """
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros((boxes1.shape[0], boxes2.shape[0]),
                               device=boxes1.device, dtype=torch.float32)

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
        Faster R-CNN style delta encoding

        Args:
            proposals: (N,4) [x1,y1,x2,y2]
            gt_boxes: (N,4) [x1,y1,x2,y2]

        Returns:
            deltas: (N,4) [dx, dy, dw, dh]
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