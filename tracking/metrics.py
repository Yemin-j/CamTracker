import torch
import motmetrics as mm
import numpy as np

def iou_matrix(gt_boxes, pred_boxes):
    """
    gt_boxes: (M,4), pred_boxes: (N,4)
    return: (M,N) IoU matrix
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)

    gt = torch.tensor(gt_boxes, dtype=torch.float32)
    pr = torch.tensor(pred_boxes, dtype=torch.float32)

    x11, y11, x12, y12 = gt[:,0], gt[:,1], gt[:,2], gt[:,3]
    x21, y21, x22, y22 = pr[:,0], pr[:,1], pr[:,2], pr[:,3]

    xa1 = torch.max(x11[:, None], x21[None, :])
    ya1 = torch.max(y11[:, None], y21[None, :])
    xa2 = torch.min(x12[:, None], x22[None, :])
    ya2 = torch.min(y12[:, None], y22[None, :])

    inter_w = (xa2 - xa1).clamp(min=0)
    inter_h = (ya2 - ya1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    iou = (inter / union).cpu().numpy()
    return iou

def compute_mot_metrics(gt_records, pred_records):
    """
    gt_records / pred_records: list of {
        "frame_id": int,
        "track_id": int,
        "bbox": [x1,y1,x2,y2],
    }
    """
    acc = mm.MOTAccumulator(auto_id=True)

    # frame 단위로 묶기
    max_frame = max(
        max((r["frame_id"] for r in gt_records), default=0),
        max((r["frame_id"] for r in pred_records), default=0)
    )

    for t in range(max_frame + 1):
        gt_t = [r for r in gt_records if r["frame_id"] == t]
        pr_t = [r for r in pred_records if r["frame_id"] == t]

        gt_ids = [r["track_id"] for r in gt_t]
        pr_ids = [r["track_id"] for r in pr_t]
        gt_boxes = [r["bbox"] for r in gt_t]
        pr_boxes = [r["bbox"] for r in pr_t]

        if len(gt_boxes) == 0 and len(pr_boxes) == 0:
            continue

        # distance matrix = 1 - IoU
        iou = iou_matrix(gt_boxes, pr_boxes)
        dist = 1 - iou  # motmetrics는 "작을수록 좋음"라서 1-IOU

        acc.update(
            gt_ids,
            pr_ids,
            dist
        )

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=mm.metrics.motchallenge_metrics,
        name='acc'
    )
    # summary는 pandas DataFrame
    return summary
