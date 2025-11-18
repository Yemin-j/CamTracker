# tracking/validation_multi.py

import torch
from tracking.single_camera import SingleCameraTracker
from tracking.tracklet_builder import TrackletBuilder
from tracking.feature_bank import TrackletFeatureExtractor
from tracking.mcta import MultiCameraAssociator
from tracking.global_id_mapper import GlobalIDMapper
from tracking.metrics import compute_mot_metrics

@torch.no_grad()
def validate_multi_camera(model, val_loader, device):

    model.eval()

    trackers = {}
    pred_records = []
    gt_records = []

    # -------------------------------
    # 1) Single-camera tracking inference
    # -------------------------------
    for frames, boxes_gt, labels_gt, tids_gt, pids, metas in val_loader:
        frames = frames.to(device)
        meta = metas[0]
        scenario = meta["scenario"]
        cam_id   = meta["cam_id"]
        frame_id = int(meta["frame_id"])

        # Save GT
        for box, tid in zip(boxes_gt[0], tids_gt[0]):
            x1,y1,x2,y2 = box.tolist()
            gt_records.append(dict(
                scenario=scenario,
                cam_id=cam_id,
                frame_id=frame_id,
                track_id=int(tid),
                bbox=[x1,y1,x2,y2],
            ))

        # Tracker
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

        for r in results:
            pred_records.append(dict(
                scenario=scenario,
                cam_id=cam_id,
                frame_id=frame_id,
                track_id=int(r["track_id"]),
                bbox=r["bbox"].cpu().tolist(),
                embed=r["embed"].cpu().tolist()
            ))

    # -------------------------------
    # 2) Build tracklets
    # -------------------------------
    builder = TrackletBuilder()
    tracklets = builder.build(pred_records)

    # 3) Extract embedding
    extractor = TrackletFeatureExtractor()
    extractor.extract(tracklets)

    # -------------------------------
    # 4) MCTA
    # -------------------------------
    associator = MultiCameraAssociator(mode="greedy", thresh=0.5)
    clusters = associator.associate(tracklets)

    mapper = GlobalIDMapper()
    mapping = mapper.build_mapping(clusters)

    pred_global = mapper.apply(pred_records, mapping)

    # -------------------------------
    # 5) Compute multi-camera MOT metrics
    # -------------------------------
    summary = compute_mot_metrics(gt_records, pred_global)
    return summary
