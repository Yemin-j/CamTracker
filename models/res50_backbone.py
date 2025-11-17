import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = resnet50(pretrained=pretrained)

        # ResNet50 stages
        self.stage1 = nn.Sequential(base.conv1, base.bn1, base.relu,
                                    base.maxpool, base.layer1)
        self.stage2 = base.layer2
        self.stage3 = base.layer3
        self.stage4 = base.layer4

    def forward(self, x):
        c1 = self.stage1(x)   # stride 4
        c2 = self.stage2(c1)  # stride 8
        c3 = self.stage3(c2)  # stride 16
        c4 = self.stage4(c3)  # stride 32
        return [c1, c2, c3, c4]

class FPN(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, 1)
            for i in range(len(in_channels))
        ])
        self.output = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(len(in_channels))
        ])

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        p4 = self.lateral[3](c4)
        p3 = self.lateral[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        p1 = self.lateral[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode='nearest')

        p4 = self.output[3](p4)
        p3 = self.output[2](p3)
        p2 = self.output[1](p2)
        p1 = self.output[0](p1)

        return [p1, p2, p3, p4]   # lowest → highest resolution

class FPNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = ResNet50Backbone()
        self.fpn  = FPN()

    def forward(self, x):
        c_feats = self.body(x)
        p_feats = self.fpn(c_feats)
        return p_feats