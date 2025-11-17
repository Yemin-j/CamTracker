# tracking/eval.py
import torch
from .metrics import compute_mot_metrics

@torch.no_grad()
def validate_tracking(model, val_loader, tracker_cls, device):
    model.eval()

    trackers = {}
    pred_records = []
    gt_records = []

    for batch in val_loader:
        frame, boxes_gt, labels_gt, track_ids_gt, pids, meta = batch
        # batch_size=1 가정
        frame = frame.to(device)
        boxes_gt = boxes_gt[0].cpu().numpy()
        track_ids_gt_ = track_ids_gt[0].cpu().numpy()
        frame_idx = int(meta["frame_id"][0])
        cam_id = meta["cam_id"][0]  # 필요하면 cam별로 별도 acc 만들 수도 있음
        scenario = meta["scenario"][0]

        # GT 기록 저장
        for b, tid in zip(boxes_gt, track_ids_gt_):
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
        outputs = model(frame)
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
    print(summary)
    model.train()
    return summary
