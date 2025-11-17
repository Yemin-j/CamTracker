import torch

from models.res50_backbone import FPNBackbone
from models.rpn_head import RPNHead
from models.roi import RoIAlignLayer
from models.bbox_head import BBoxHead
from models.reid_head import ReIDHead
from models.detector import DetectorWithReID


def build_detector():
    backbone = FPNBackbone()
    rpn_head = RPNHead(
        in_channels=256,
        num_anchors=9,
        strides=[8,16,32,64],
        scales=[4,8,16],
        ratios=[0.5,1.0,2.0],
        pre_nms_topk=200,
        post_nms_topk=50,
        nms_thresh=0.7
    )
    roi_align = RoIAlignLayer(output_size=7)
    bbox_head = BBoxHead(in_channels=256, fc_dim=1024, num_classes=1)
    reid_head = ReIDHead(in_channels=256, embed_dim=256)

    return DetectorWithReID(backbone, rpn_head, roi_align, bbox_head, reid_head)


if __name__ == "__main__":
    # 1) dummy image 준비
    img = torch.randn(1, 3, 640, 640)  # B=1, C=3, H=640, W=640

    # 2) 모델 만들기
    model = build_detector()
    model.eval()

    # 3) forward 돌리기
    with torch.no_grad():
        outputs = model(img)

    print("==== Forward Output ====")
    print(outputs)

    for out in outputs:
        print("boxes:", out["boxes"].shape)
        print("scores:", out["scores"].shape)
        print("labels:", out["labels"].shape)
        print("embeds:", out["embeds"].shape)
