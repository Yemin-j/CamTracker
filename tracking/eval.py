# tracking/eval.py
import torch
from .metrics import compute_mot_metrics

@torch.no_grad()
def validate_tracking(model, val_loader, tracker_cls, device):
    model.eval()

    trackers = {}
    pred_records = []
    gt_records = []

    for frames, boxes_gt, labels_gt, tids_gt, pids, metas in val_loader:
        # batch_size=1 가정
        frames = frames.to(device)

        # metas는 list, 그 안에 dict 하나
        meta = metas[0]
        cam_id   = meta["cam_id"]
        scenario = meta["scenario"]

        # frame_id는 이미 int이거나 0-dim tensor일 가능성
        fid = meta["frame_id"]
        if hasattr(fid, "item"):   # tensor일 때
            frame_idx = int(fid.item())
        else:
            frame_idx = int(fid)

        # GT 기록 저장 (boxes_gt[0], tids_gt[0] : 해당 프레임의 GT)
        boxes_np = boxes_gt[0].cpu().numpy()
        tids_np  = tids_gt[0].cpu().numpy()

        for b, tid in zip(boxes_np, tids_np):
            x1, y1, x2, y2 = b.tolist()
            gt_records.append(dict(
                frame_id=frame_idx,
                track_id=int(tid),
                bbox=[x1, y1, x2, y2],
                cam_id=cam_id,
                scenario=scenario
            ))

        # tracker 준비
        if cam_id not in trackers:
            trackers[cam_id] = tracker_cls(
                max_age=30,
                sim_thresh=0.0,
                iou_thresh=0.0,
                alpha=1.0
            )

        # 모델 inference
        outputs = model(frames)
        out = outputs[0]
        det_boxes  = out["boxes"].to(device)
        det_scores = out["scores"].to(device)
        det_labels = out["labels"].to(device)
        det_embeds = out["embeds"].to(device)

        # tracking update
        results = trackers[cam_id].update(
            det_boxes, det_scores, det_labels, det_embeds, frame_idx
        )

        # 예측 기록 저장
        for r in results:
            x1, y1, x2, y2 = r["bbox"].cpu().tolist()
            pred_records.append(dict(
                frame_id=frame_idx,
                track_id=int(r["track_id"]),
                bbox=[x1, y1, x2, y2],
                cam_id=cam_id,
                scenario=scenario
            ))

    # 이제 gt_records vs pred_records 로 MOT metrics 계산
    summary = compute_mot_metrics(gt_records, pred_records)
    model.train()
    return summary
