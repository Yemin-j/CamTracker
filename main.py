import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import yaml
from torch.utils.data import DataLoader
from datasets.mot_dataloader import AIMTMDCVideoDataset
from models.res50_backbone import FPNBackbone
from models.rpn_head import RPNHead
from models.roi import RoIAlignLayer
from models.bbox_head import BBoxHead
from models.reid_head import ReIDHead
from models.detector import DetectorWithReID
from tracking.single_camera import SingleCameraTracker
from tracking.multi_camera import multi_cam_association

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_detector_with_reid(device):
    backbone = FPNBackbone()
    rpn_head = RPNHead(
        in_channels=256,
        num_anchors=9,          # scales(3) × ratios(3)
        strides=[8,16,32,64],
        scales=[4,8,16],
        ratios=[0.5,1.0,2.0],
        pre_nms_topk=1000,
        post_nms_topk=200,
        nms_thresh=0.7
    )
    roi_align = RoIAlignLayer(output_size=7)
    bbox_head = BBoxHead(in_channels=256, fc_dim=1024, num_classes=1, score_thresh=0.0,)
    reid_head = ReIDHead(in_channels=256, embed_dim=256)

    model = DetectorWithReID(backbone, rpn_head, roi_align, bbox_head, reid_head)
    model.to(device)
    model.eval()
    return model

def build_dataloader(cfg):
    dataset = AIMTMDCVideoDataset(
        video_root=cfg["data"]["video_root"],
        ann_root=cfg["data"]["ann_root"],
        scenario_ids=cfg["data"]["scenario_ids"],
        transform=None  # TODO: augment/normalize 추가
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["train"]["shuffle"],
        num_workers=cfg["train"]["num_workers"]
    )
    return loader

def build_model(cfg):
    # TODO: backbone/roi_head/reid_head 구성해서 넣기
    backbone = ...
    roi_head = ...
    reid_head = ...
    model = DetectorWithReID(backbone, roi_head, reid_head)
    return model

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Dataset & DataLoader
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    video_root = r"D:/tar_trac/data/videos/train"
    ann_root   = r"D:/tar_trac/data/annotations/train"

    scenario_ids = [f"s{i:02d}" for i in range(1, 2)]

    dataset = AIMTMDCVideoDataset(
        video_root=video_root,
        ann_root=ann_root,
        scenario_ids=scenario_ids,
        transform=transform
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2) Model
    model = build_detector_with_reid(device)

    # 3) Camera별 Tracker 준비
    trackers = {}        # cam_id -> SingleCameraTracker
    all_tracks = []      # [{scenario, cam_id, frame_id, track_id, bbox}, ...]

    with torch.no_grad():
        for frame, boxes_gt, labels_gt, track_ids_gt, pids, meta in loader:
            # batch_size=1 가정
            frame = frame.to(device)
            cam_id = meta["cam_id"][0]
            frame_idx = int(meta["frame_id"][0])
            scenario = meta["scenario"][0]

            if cam_id not in trackers:
                trackers[cam_id] = SingleCameraTracker(
                    max_age=30,
                    sim_thresh=0.0,
                    iou_thresh=0.0,
                    alpha=1.0
                )

            # 4) detector forward
            outputs = model(frame)
            out = outputs[0]
            det_boxes  = out["boxes"].to(device)
            det_scores = out["scores"].to(device)
            det_labels = out["labels"].to(device)
            det_embeds = out["embeds"].to(device)

            # 5) tracking update (per-camera)
            results = trackers[cam_id].update(
                det_boxes, det_scores, det_labels, det_embeds, frame_idx
            )

            print(f"[frame {frame_idx}] boxes={out['boxes'].shape[0]}, "
                  f"tracks={len(results)}, max_score={out['scores'].max().item():.3f}")

            for r in results:
                all_tracks.append(dict(
                    scenario=scenario,
                    cam_id=cam_id,
                    frame_id=frame_idx,
                    track_id=r["track_id"],
                    bbox=r["bbox"].cpu().tolist()
                ))

    # 6) 결과 확인 (일부만 출력)
    print("Total track records:", len(all_tracks))
    for r in all_tracks[:20]:
        print(r)

    # cfg = load_config("configs/kaist_mtmdc.yaml")
    #
    # # 1) 데이터 로더
    # loader = build_dataloader(cfg)
    #
    # # 2) 모델
    # model = build_model(cfg)
    # model.to(cfg["train"]["device"])
    # model.train()  # or eval for inference
    #
    # # 3) 카메라별 트래커 준비
    # trackers = {}
    # all_track_records = []
    #
    # for frame, boxes, labels, track_ids, pids, meta in loader:
    #     cam = meta["cam_id"][0]   # batch_size=1 가정
    #     frame_idx = int(meta["frame_id"][0])
    #
    #     if cam not in trackers:
    #         trackers[cam] = SingleCameraTracker(
    #             max_age=cfg["track"]["max_age"],
    #             match_thresh=cfg["track"]["match_thresh"]
    #         )
    #
    #     # TODO: model forward (det+embed) 해서 boxes/scores/labels/embeds 얻기
    #     # 여기서는 annotation을 대신 사용중이라면 생략 가능
    #
    #     # 추적 업데이트
    #     results = trackers[cam].update(
    #         det_boxes=boxes[0],
    #         det_scores=None,  # placeholder
    #         det_labels=labels[0],
    #         det_embeds=None,  # placeholder
    #         frame_idx=frame_idx
    #     )
    #     # all_track_records에 누적 (multicam 연계용)
    #
    # # 4) 트랙 단위로 평균 임베딩 계산 + 멀티카메라 association
    # # multi_cam_association(...)

if __name__ == "__main__":
    run()
