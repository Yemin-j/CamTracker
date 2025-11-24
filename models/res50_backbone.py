import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 backbone for feature extraction

    Outputs 4 feature maps at different scales:
    - C1: stride 4  (after conv1, maxpool, layer1)
    - C2: stride 8  (after layer2)
    - C3: stride 16 (after layer3)
    - C4: stride 32 (after layer4)
    """

    def __init__(self, pretrained=True):
        super().__init__()
        base = resnet50(pretrained=pretrained)

        # ResNet50 stages
        self.stage1 = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1
        )
        self.stage2 = base.layer2
        self.stage3 = base.layer3
        self.stage4 = base.layer4

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images

        Returns:
            List of feature maps [C1, C2, C3, C4]
            - C1: (B, 256, H/4, W/4)
            - C2: (B, 512, H/8, W/8)
            - C3: (B, 1024, H/16, W/16)
            - C4: (B, 2048, H/32, W/32)
        """
        c1 = self.stage1(x)  # stride 4,  256 channels
        c2 = self.stage2(c1)  # stride 8,  512 channels
        c3 = self.stage3(c2)  # stride 16, 1024 channels
        c4 = self.stage4(c3)  # stride 32, 2048 channels
        return [c1, c2, c3, c4]


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN)

    Builds top-down feature pyramid with lateral connections
    """

    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=256):
        """
        Args:
            in_channels: Channel dimensions for [C1, C2, C3, C4]
            out_channels: Output channel dimension for all pyramid levels
        """
        super().__init__()

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            for i in range(len(in_channels))
        ])

        # Output convolutions (3x3 conv to reduce aliasing)
        self.output = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels))
        ])

    def forward(self, feats):
        """
        Args:
            feats: List of backbone features [C1, C2, C3, C4]

        Returns:
            List of FPN features [P2, P3, P4, P5]
            - P2: stride 4  (from C1)
            - P3: stride 8  (from C2)
            - P4: stride 16 (from C3)
            - P5: stride 32 (from C4)

            All have same channel dimension (256 by default)
        """
        c1, c2, c3, c4 = feats

        # Top-down pathway with lateral connections
        p4 = self.lateral[3](c4)
        p3 = self.lateral[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        p1 = self.lateral[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode='nearest')

        # Apply output convolutions
        p4 = self.output[3](p4)
        p3 = self.output[2](p3)
        p2 = self.output[1](p2)
        p1 = self.output[0](p1)

        # Return in order: [P2, P3, P4, P5]
        # P2 = highest resolution (stride 4)
        # P5 = lowest resolution (stride 32)
        return [p1, p2, p3, p4]


class FPNBackbone(nn.Module):
    """
    Complete FPN backbone: ResNet-50 + FPN

    This is the standard backbone for QDTrack and Faster R-CNN
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained: Whether to use pretrained ResNet-50 weights
        """
        super().__init__()
        self.body = ResNet50Backbone(pretrained=pretrained)
        self.fpn = FPN()

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images

        Returns:
            List of FPN features [P2, P3, P4, P5]
            Each has shape (B, 256, H_i, W_i)
            - P2: stride 4  (H/4, W/4)
            - P3: stride 8  (H/8, W/8)
            - P4: stride 16 (H/16, W/16)
            - P5: stride 32 (H/32, W/32)
        """
        c_feats = self.body(x)
        p_feats = self.fpn(c_feats)
        return p_feats