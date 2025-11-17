from .res50_backbone import FPNBackbone
from .rpn_head import RPNHead
from .roi import RoIAlignLayer
from .bbox_head import BBoxHead
from .reid_head import ReIDHead
from .detector import DetectorWithReID

def build_detector_with_reid():
    backbone = FPNBackbone()
    rpn_head = RPNHead()
    roi_align = RoIAlignLayer()
    bbox_head = BBoxHead()
    reid_head = ReIDHead()
    return DetectorWithReID(backbone, rpn_head, roi_align, bbox_head, reid_head)
