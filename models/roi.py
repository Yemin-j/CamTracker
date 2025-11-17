import torch.nn as nn
from torchvision.ops import roi_align

class RoIAlignLayer(nn.Module):
    def __init__(self, output_size=7):
        super().__init__()
        self.output_size = output_size

    def forward(self, feats, proposals):
        # proposals: (N,4) absolute coordinates
        # feats: list of P2..P5
        # 간단 버전: 가장 해상도 높은 P2에만 ROIAlign
        return roi_align(
            input=feats[0],  # P2: highest resolution
            boxes=[proposals],
            output_size=self.output_size,
            spatial_scale=1.0/4.0,
            sampling_ratio=2
        )
