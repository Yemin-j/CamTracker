import torch

class AnchorGenerator:
    """
    PyTorch에서 쉽게 anchor grid를 만드는 클래스.
    FPN 레벨마다 stride / scales / ratios를 지정할 수 있음.
    """

    def __init__(self,
                 strides=[8, 16, 32, 64],
                 scales=[32, 64, 128],
                 ratios=[0.5, 1.0, 2.0]):
        """
        strides: 각 FPN 레벨의 stride 값
        scales:  anchor base 크기
        ratios:  height/width 비율
        """
        self.strides = strides
        self.scales = scales
        self.ratios = ratios

        # pre-calc anchor templates per level
        self.base_anchors = self._generate_base_anchors()

    def _generate_base_anchors(self):
        base_anchors = []
        for stride in self.strides:
            anchors = []
            for scale in self.scales:
                for ratio in self.ratios:
                    size = scale * stride
                    size = torch.tensor(size, dtype=torch.float32)
                    ratio_t = torch.tensor(ratio, dtype=torch.float32)

                    w = size * torch.sqrt(1.0 / ratio_t)
                    h = size * torch.sqrt(ratio_t)

                    anchors.append([
                        -0.5 * w, -0.5 * h, 0.5 * w, 0.5 * h
                    ])

            anchors = torch.stack([torch.tensor(a, dtype=torch.float32).flatten() for a in anchors])
            base_anchors.append(anchors)
        return base_anchors

    def grid_anchors(self, feat_shapes):
        """
        feat_shapes: [(H1,W1), (H2,W2), ...] FPN level별 feature map shape
        return: anchors_per_level = [Tensor(shape=(Hi*Wi*A, 4)), ...]
        """

        anchors_per_level = []

        for (H, W), stride, base_anchor in zip(feat_shapes, self.strides, self.base_anchors):
            # grid 좌표 생성
            shift_x = torch.arange(W) * stride
            shift_y = torch.arange(H) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)  # (H*W,4)

            A = base_anchor.shape[0]  # anchor 수
            K = shifts.shape[0]       # grid cell 수
            anchors = (base_anchor.view(1, A, 4) + shifts.view(K, 1, 4)).reshape(-1,4)
            anchors_per_level.append(anchors)

        return anchors_per_level
