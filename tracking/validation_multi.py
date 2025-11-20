# tracking/validation_multi.py
import torch
import time
from tracking.single_camera import SingleCameraTracker
from tracking.tracklet_builder import TrackletBuilder
from tracking.feature_bank import TrackletFeatureExtractor
from tracking.mcta import MultiCameraAssociator
from tracking.global_id_mapper import GlobalIDMapper
from tracking.metrics import compute_mot_metrics

from utils.export_utils import save_single_camera_avi # for test

@torch.no_grad()
def validate_multi_camera(model, val_loader, device, logger=None, export_per_cam=False, video_root=None):
    """
    Single pass:
      - run single-camera tracking for all cameras
      - build GT / pred_records
      - compute single-camera MOT metrics
      - build tracklets and do MCTA
      - compute multi-camera MOT metrics

    return:
      summary_single: DataFrame (single-camera MOT metrics)
      summary_mcta  : DataFrame (multi-camera/global MOT metrics)
      pred_records  : list[dict], single-camera tracking 결과
      pred_global   : list[dict], multi-camera/global tracking 결과
    """

    model.eval()

    trackers = {}
    pred_records = []
    gt_records = []

    # --------------------------------------------------
    # 0) 카메라별 총 프레임 수 계산
    # --------------------------------------------------

    cam_total_frames = {}
    ds = val_loader.dataset  # AIMTMDCVideoDataset
    for item in ds.index:
        key = (item["scenario"], item["cam_id"])
        cam_total_frames[key] = cam_total_frames.get(key, 0) + 1

    cam_processed = {}
    cam_start_time = {}

    ##### for test
    current_key = None  # (scenario, cam_id)
    cam_records = []  # 현재 카메라의 pred_records만 모으는 리스트
    #####

    # -------------------------------
    # 1) Single-camera tracking inference (모델 forward 1회)
    # -------------------------------
    for frames, boxes_gt, labels_gt, tids_gt, pids, metas in val_loader:
        frames = frames.to(device)
        meta = metas[0]
        scenario = meta["scenario"]
        cam_id   = meta["cam_id"]
        frame_id = int(meta["frame_id"])

        key = (scenario, cam_id)

        if key not in cam_processed:
            cam_processed[key] = 0
            cam_start_time[key] = time.time()

        cam_processed[key] += 1

        total = cam_total_frames[key]
        done = cam_processed[key]
        progress = 100.0 * done / total

        elapsed = time.time() - cam_start_time[key]
        avg_t = elapsed / max(done, 1)
        remaining = total - done
        eta_sec = remaining * avg_t
        eta_min = int(eta_sec // 60)
        eta_sec = int(eta_sec % 60)

        if logger and (done % 40 == 0 or done == total):
            logger.info(
                f"[VALID] {scenario}/{cam_id} frame={frame_id} "
                f"[{done}/{total} {progress:.1f}%] ETA {eta_min}m {eta_sec}s"
            )

        # GT 기록
        for box, tid in zip(boxes_gt[0], tids_gt[0]):
            x1, y1, x2, y2 = box.tolist()
            gt_records.append(dict(
                scenario=scenario,
                cam_id=cam_id,
                frame_id=frame_id,
                track_id=int(tid),
                bbox=[x1, y1, x2, y2],
            ))

        # Tracker 준비
        if cam_id not in trackers:
            trackers[cam_id] = SingleCameraTracker()

        out = model(frames)[0]
        results = trackers[cam_id].update(
            out["boxes"].to(device),
            out["scores"].to(device),
            out["labels"].to(device),
            out["embeds"].to(device),
            frame_id
        )

        # 단일 카메라 결과 저장
        for r in results:
            rec = dict(
                scenario=scenario,
                cam_id=cam_id,
                frame_id=frame_id,
                track_id=int(r["track_id"]),
                bbox=r["bbox"].cpu().tolist(),
                embed=r["embed"].cpu().tolist()
            )
            pred_records.append(rec)
            # cam_records.append(rec)

    # -------------------------------
    # 2) SINGLE-CAMERA METRICS
    # -------------------------------
    summary_single = compute_mot_metrics(gt_records, pred_records)

    # -------------------------------
    # 3) Build tracklets for MCTA
    # -------------------------------
    builder = TrackletBuilder()
    tracklets = builder.build(pred_records)

    # 4) tracklet embedding 추출
    extractor = TrackletFeatureExtractor()
    extractor.extract(tracklets)

    # -------------------------------
    # 5) MCTA association
    # -------------------------------
    associator = MultiCameraAssociator(mode="greedy", thresh=0.5)
    clusters = associator.associate(tracklets)

    mapper = GlobalIDMapper()
    mapping = mapper.build_mapping(clusters)
    pred_global = mapper.apply(pred_records, mapping)

    # -------------------------------
    # 6) MULTI-CAMERA METRICS
    # -------------------------------
    summary_mcta = compute_mot_metrics(gt_records, pred_global)

    model.train()  # train 모드 복원

    return summary_single, summary_mcta, pred_records, pred_global
