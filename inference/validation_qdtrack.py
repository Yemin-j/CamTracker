import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from utils.export_utils import save_single_camera_avi


def validate_qdtrack(model, val_loader, device, logger=None, max_batches=None):
    """
    Validate QDTrack model on validation set.

    Args:
        model: QDTrackDetector model
        val_loader: DataLoader for validation (temporal pair format, same as train)
                    batch format (from QDTrackCollateFn):
                      (frames_key, frames_ref,
                       boxes_key, boxes_ref,
                       ids_key, ids_ref,
                       metas)
        device: Device to run on
        logger: Logger instance
        max_batches: Maximum batches to validate (None = all)

    Returns:
        dict with validation metrics, including detection_results list
    """
    model.eval()

    if logger:
        logger.info("Starting validation...")

    total_detections = 0
    total_gt_boxes = 0
    detection_results = []

    with torch.no_grad():
        iterator = enumerate(val_loader)
        if logger:
            iterator = tqdm(iterator, total=len(val_loader), desc="Validation", ncols=100, leave=True)

        for batch_idx, batch in iterator:
            if max_batches is not None and batch_idx >= max_batches:
                break

            # --- batch unpack ---
            # temporal pair format from QDTrackCollateFn
            if len(batch) == 7:
                frames_key, frames_ref, boxes_key, boxes_ref, ids_key, ids_ref, metas = batch
                frames = frames_key  # validation에서는 key frame만 사용
                boxes = boxes_key
            else:
                # 혹시 단일 프레임 포맷을 쓰는 경우를 위한 fallback
                frames, boxes, labels, tids, pids, metas = batch

            frames = frames.to(device, non_blocking=True)

            # --- forward inference ---
            outputs = model(frames)  # List[dict] per image

            # --- per-image 처리 ---
            for img_idx, (output, meta) in enumerate(zip(outputs, metas)):
                pred_boxes = output.get("boxes", torch.empty((0, 4), device=device)).detach().cpu()
                pred_scores = output.get("scores", torch.empty((0,), device=device)).detach().cpu()
                pred_labels = output.get("labels", torch.empty((0,), device=device)).detach().cpu()
                # embeds는 tracking 용이니 validation metric에는 사용 안 함
                pred_embeds = output.get("embeds", torch.empty((0, 256), device=device)).detach().cpu()

                gt_boxes_img = boxes[img_idx].cpu() if len(boxes) > 0 else torch.empty((0, 4))

                total_detections += len(pred_boxes)
                total_gt_boxes += len(gt_boxes_img)

                # frame id / seq id 추출 (VisDrone용)
                # visdrone_dataloader meta: {"seq": seq, "key_frame": kfid, "ref_frame": rfid}
                seq_id = meta.get("seq", "unknown")
                frame_id = None
                if "frame_id" in meta:
                    frame_id = meta["frame_id"]
                elif "key_frame" in meta:
                    frame_id = meta["key_frame"]
                elif "key_frame_id" in meta:
                    frame_id = meta["key_frame_id"]
                else:
                    frame_id = 0

                detection_results.append({
                    "scenario": seq_id,
                    "cam_id": seq_id,  # VisDrone에서는 seq를 cam_id처럼 사용
                    "frame_id": int(frame_id),
                    "num_pred": len(pred_boxes),
                    "num_gt": len(gt_boxes_img),
                    "pred_boxes": pred_boxes.numpy(),
                    "pred_scores": pred_scores.numpy(),
                    "gt_boxes": gt_boxes_img.numpy(),
                })

    num_frames = len(detection_results)
    avg_det_per_frame = total_detections / num_frames if num_frames > 0 else 0.0
    avg_gt_per_frame = total_gt_boxes / num_frames if num_frames > 0 else 0.0

    metrics = {
        "total_frames": num_frames,
        "total_detections": total_detections,
        "total_gt_boxes": total_gt_boxes,
        "avg_detections_per_frame": avg_det_per_frame,
        "avg_gt_per_frame": avg_gt_per_frame,
        "detection_results": detection_results,
    }

    if logger:
        logger.info("Validation completed:")
        logger.info(f"  Total frames: {metrics['total_frames']}")
        logger.info(f"  Total detections: {metrics['total_detections']}")
        logger.info(f"  Total GT boxes: {metrics['total_gt_boxes']}")
        logger.info(f"  Avg detections/frame: {metrics['avg_detections_per_frame']:.2f}")
        logger.info(f"  Avg GT/frame: {metrics['avg_gt_per_frame']:.2f}")

    model.train()
    return metrics


def build_validation_dataloader(cfg, device):
    """
    Build validation dataloader for VisDrone2019-MOT.

    - train과 동일한 VisDroneQDTrackDataset(temporal pair) 사용
    - shuffle=False 만 다르게 설정
    - person-only filter 등은 dataset 쪽에서 이미 적용
    """
    from torch.utils.data import DataLoader
    from datasets.visdrone_dataloader import VisDroneQDTrackDataset
    from datasets.mot_collate_fn import QDTrackCollateFn
    from datasets.load_ids import list_visdrone_sequences

    val_root = cfg["val"]["video_root"]
    ann_root = cfg["val"]["ann_root"]

    # seq id 자동 스캔
    val_ids = list_visdrone_sequences(val_root)

    temporal_dataset = VisDroneQDTrackDataset(
        video_root=val_root,
        ann_root=ann_root,
        scenario_ids=val_ids,
        frame_stride=cfg["hyperparams"].get("val_frame_stride", 1),
        temporal_offset_range=cfg["hyperparams"].get("val_temporal_offset", 2),
        transform=None,
    )

    # train과 같은 collate, shuffle만 False
    val_loader = DataLoader(
        temporal_dataset,
        batch_size=cfg["hyperparams"].get("val_batch_size", 1),
        shuffle=False,
        num_workers=cfg["hyperparams"].get("val_num_workers", 0),
        collate_fn=QDTrackCollateFn(),
        pin_memory=True,
    )

    return val_loader


def compute_detection_metrics(detection_results, iou_threshold=0.5):
    """
    Compute detection metrics (Precision, Recall, F1)
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for result in detection_results:
        pred_boxes = result["pred_boxes"]
        gt_boxes = result["gt_boxes"]

        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue

        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue

        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        ious = compute_iou_matrix(
            torch.from_numpy(pred_boxes),
            torch.from_numpy(gt_boxes),
        ).numpy()

        matched_gt = set()
        tp = 0

        for pred_idx in range(len(pred_boxes)):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                if ious[pred_idx, gt_idx] > best_iou:
                    best_iou = ious[pred_idx, gt_idx]
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)

        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def compute_iou_matrix(boxes1, boxes2):
    """
    Compute IoU matrix between two sets of boxes

    Args:
        boxes1: (N, 4) tensor [x1, y1, x2, y2]
        boxes2: (M, 4) tensor [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)

    wh = (rb - lt).clamp(min=0)  # (N,M,2)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    return iou
