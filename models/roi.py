import torch
import torch.nn as nn
from torchvision.ops import roi_align
import math


class RoIAlignLayer(nn.Module):
    """
    FPN-based RoI Align Layer

    Automatically assigns RoIs to appropriate FPN levels based on box size.
    Supports batch processing.
    """

    def __init__(self, output_size=7, canonical_box_size=224, canonical_level=2):
        """
        Args:
            output_size: Output feature map size (e.g., 7 for 7x7)
            canonical_box_size: Reference box size for level assignment
            canonical_level: FPN level for canonical box size (2 for P2)
        """
        super().__init__()
        self.output_size = output_size
        self.canonical_box_size = canonical_box_size
        self.canonical_level = canonical_level

    def forward(self, feats, proposals):
        """
        Apply RoI Align on FPN features

        Args:
            feats: List of FPN features [P2, P3, P4, P5]
                   Each has shape (B, C, H, W)
            proposals: Proposals for ONE image, shape (N, 4) in xyxy format

        Returns:
            roi_features: (N, C, output_size, output_size)
        """
        if proposals.numel() == 0:
            # No proposals - return empty tensor
            device = feats[0].device
            C = feats[0].shape[1]
            return torch.empty((0, C, self.output_size, self.output_size),
                               device=device, dtype=torch.float32)

        # FPN levels and their strides
        # feats = [P2, P3, P4, P5]
        # strides = [4, 8, 16, 32]
        num_levels = len(feats)
        strides = [2 ** (i + 2) for i in range(num_levels)]  # [4, 8, 16, 32]

        # Assign each proposal to appropriate FPN level
        target_levels = self._assign_boxes_to_levels(
            proposals, strides, self.canonical_box_size, self.canonical_level
        )

        # Collect RoI features from each level
        roi_features_list = []

        for level_idx in range(num_levels):
            # Find proposals assigned to this level
            level_mask = target_levels == level_idx

            if level_mask.sum() == 0:
                continue

            level_proposals = proposals[level_mask]

            # RoI Align on this level
            # Note: roi_align expects boxes as list of tensors (one per image)
            # Since we're processing one image, wrap in list
            level_features = roi_align(
                input=feats[level_idx],
                boxes=[level_proposals],  # List of [proposals_for_img0, ...]
                output_size=self.output_size,
                spatial_scale=1.0 / strides[level_idx],
                sampling_ratio=2,
                aligned=True  # Use aligned=True for better accuracy
            )

            roi_features_list.append((level_mask, level_features))

        # Merge features back to original order
        roi_features = self._merge_features(roi_features_list, proposals.shape[0], feats[0])

        return roi_features

    @staticmethod
    def _assign_boxes_to_levels(boxes, strides, canonical_box_size, canonical_level):
        """
        Assign boxes to FPN levels based on box area

        Uses the formula from FPN paper:
        k = floor(k0 + log2(sqrt(wh) / s0))

        Args:
            boxes: (N, 4) boxes in xyxy format
            strides: List of strides [4, 8, 16, 32]
            canonical_box_size: Reference box size (224)
            canonical_level: Reference level (2 for P2)

        Returns:
            target_levels: (N,) level indices [0, 1, 2, 3] for [P2, P3, P4, P5]
        """
        # Compute box areas
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights

        # Compute target level
        # k = k0 + log2(sqrt(area) / s0)
        sqrt_areas = torch.sqrt(areas.clamp(min=1.0))
        target_levels = torch.floor(
            canonical_level + torch.log2(sqrt_areas / canonical_box_size)
        )

        # Clamp to valid level range [0, len(strides)-1]
        target_levels = target_levels.clamp(min=0, max=len(strides) - 1).long()

        return target_levels

    @staticmethod
    def _merge_features(roi_features_list, num_proposals, reference_feat):
        """
        Merge RoI features from different levels back to original order

        Args:
            roi_features_list: List of (mask, features) tuples
            num_proposals: Total number of proposals
            reference_feat: Reference feature map for getting device/dtype

        Returns:
            merged_features: (N, C, H, W) in original proposal order
        """
        device = reference_feat.device
        C = reference_feat.shape[1]
        output_size = roi_features_list[0][1].shape[-1]

        # Initialize output tensor
        merged = torch.zeros(
            (num_proposals, C, output_size, output_size),
            device=device,
            dtype=reference_feat.dtype
        )

        # Fill in features from each level
        for level_mask, level_features in roi_features_list:
            merged[level_mask] = level_features

        return merged


class SimplifiedRoIAlignLayer(nn.Module):
    """
    Simplified RoI Align that only uses P2 (stride=4)

    Use this if you want to match the original simple implementation
    or for faster training/debugging.
    """

    def __init__(self, output_size=7):
        super().__init__()
        self.output_size = output_size

    def forward(self, feats, proposals):
        """
        Args:
            feats: List of FPN features [P2, P3, P4, P5]
            proposals: (N, 4) proposals in xyxy format

        Returns:
            roi_features: (N, C, output_size, output_size)
        """
        if proposals.numel() == 0:
            device = feats[0].device
            C = feats[0].shape[1]
            return torch.empty((0, C, self.output_size, self.output_size),
                               device=device, dtype=torch.float32)

        # Use only P2 (stride=4)
        return roi_align(
            input=feats[0],  # P2: stride=4
            boxes=[proposals],
            output_size=self.output_size,
            spatial_scale=1.0 / 4.0,
            sampling_ratio=2,
            aligned=True
        )