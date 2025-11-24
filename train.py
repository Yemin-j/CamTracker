import os
import yaml
import torch
import time
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

# QDTrack Dataset & Collate
from datasets.visdrone_dataloader import VisDroneQDTrackDataset
from datasets.mot_collate_fn import QDTrackCollateFn
from datasets.load_ids import list_visdrone_sequences

# model
from models.detector import QDTrackDetector

# validation
from inference.validation_qdtrack import validate_qdtrack, build_validation_dataloader, compute_detection_metrics
from tracking.validation_multi import validate_multi_camera

# Utils
from utils.logger import setup_logger
from utils.export_utils import save_visdrone_video
from utils.checkpoint import CheckpointManager


# -----------------------------------------------------------
# Memory & GPU Monitoring
# -----------------------------------------------------------
def log_gpu_memory(logger=None):
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3

        msg = (f"GPU Memory: Allocated={allocated:.2f}GB, "
               f"Reserved={reserved:.2f}GB, "
               f"Max={max_allocated:.2f}GB")

        if logger:
            logger.info(msg)
        else:
            print(msg)

        return allocated, reserved, max_allocated
    return 0, 0, 0


def clear_gpu_cache(logger=None):
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if logger:
            logger.info("GPU cache cleared")


# -----------------------------------------------------------
# Build QDTrack Model
# -----------------------------------------------------------
def build_qdtrack_model(cfg, device):
    """
    Build QDTrack detector with updated architecture

    Args:
        cfg: Configuration dict
        device: Device to place model on

    Returns:
        QDTrackDetector model
    """
    model = QDTrackDetector.build_default(
        # Architecture params
        in_channels=cfg["model"].get("in_channels", 256),
        fc_dim=cfg["model"].get("fc_dim", 1024),
        num_classes=cfg["model"].get("num_classes", 1),
        embed_dim=cfg["model"].get("embed_dim", 256),

        # Detection params
        score_thresh=cfg["model"].get("score_thresh", 0.05),
        nms_thresh=cfg["model"].get("nms_thresh", 0.5),
        max_dets=cfg["model"].get("max_dets", 100),

        # ReID params
        temperature=cfg["model"].get("temperature", 0.07),

        # RPN params
        rpn_pre_nms=cfg["model"].get("rpn_pre_nms", 1000),
        rpn_post_nms=cfg["model"].get("rpn_post_nms", 200),
        rpn_nms_thresh=cfg["model"].get("rpn_nms_thresh", 0.7),

        # Anchor params (P2-P5 strides)
        strides=[4, 8, 16, 32],
        scales=cfg["model"].get("anchor_scales", [4, 8, 16]),
        ratios=cfg["model"].get("anchor_ratios", [0.5, 1.0, 2.0]),

        # Backbone
        pretrained=cfg["model"].get("pretrained", True),

        device=device
    )

    return model


# -----------------------------------------------------------
# Training Function
# -----------------------------------------------------------
def train():
    # ---------------------------
    # Load Config
    # ---------------------------
    with open("./pknu_mtmdc.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # AMP (Mixed Precision) setup
    use_amp = cfg["hyperparams"].get("use_amp", True)
    scaler = GradScaler() if use_amp else None

    # Setup logger
    timestamp = str(int(time.time() * 1000))
    timestamp = 'qdtrack_train_' + timestamp
    save_root = os.path.join(cfg["log"]["save_dir"], timestamp)
    os.makedirs(save_root, exist_ok=True)
    log_file = os.path.join(save_root, "train.log")
    logger = setup_logger(log_file)

    logger.info("=" * 60)
    logger.info("QDTrack Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Mixed Precision (AMP): {use_amp}")

    # Checkpoint manager
    ckpt_manager = CheckpointManager(save_root, logger)

    # ---------------------------
    # Build QDTrack Dataset
    # ---------------------------
    train_ids = list_visdrone_sequences(cfg["train"]["video_root"])
    logger.info("Building training dataset...")

    train_dataset = VisDroneQDTrackDataset(
        video_root=cfg["train"]["video_root"],
        ann_root=cfg["train"]["ann_root"],
        scenario_ids=train_ids,
        transform=None,
        frame_stride=cfg["hyperparams"].get("frame_stride", 1),
        temporal_offset_range=cfg["hyperparams"].get("temporal_offset_range", 3),
    )

    logger.info(f"Training dataset size: {len(train_dataset)} frame pairs")

    # DataLoader with temporal pairs
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["hyperparams"]["batch_size"],
        shuffle=True,
        num_workers=cfg["hyperparams"].get("num_workers", 4),
        collate_fn=QDTrackCollateFn(),
        pin_memory=True,
        persistent_workers=True,  #
        drop_last=True  # Drop incomplete batches
    )

    logger.info(f"Batches per epoch: {len(train_loader)}")

    # ---------------------------
    # Build Validation Dataset
    # ---------------------------
    val_ids = list_visdrone_sequences(cfg["val"]["video_root"])
    logger.info("Building validation dataset...")

    val_loader = build_validation_dataloader(cfg, device)
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")

    # ---------------------------
    # Build Model & Optimizer
    # ---------------------------
    logger.info("Building QDTrack model...")
    model = build_qdtrack_model(cfg, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer (AdamW recommended for transformers/modern architectures)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["hyperparams"]["lr"],
        weight_decay=cfg["hyperparams"].get("weight_decay", 0.0001),
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler (optional but recommended)
    use_scheduler = cfg["hyperparams"].get("use_scheduler", True)
    if use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg["hyperparams"]["epochs"],
            eta_min=cfg["hyperparams"]["lr"] * 0.01
        )
    else:
        scheduler = None

    # Training params
    epochs = cfg["hyperparams"]["epochs"]
    val_interval = cfg["hyperparams"].get("val_interval", 1000)
    log_interval = cfg["hyperparams"].get("log_interval", 50)

    # QDTrack specific params
    pos_iou_thr = cfg["hyperparams"].get("pos_iou_thr", 0.7)
    neg_iou_thr = cfg["hyperparams"].get("neg_iou_thr", 0.3)
    num_samples_key = cfg["hyperparams"].get("num_samples_key", 128)
    num_samples_ref = cfg["hyperparams"].get("num_samples_ref", 256)

    total_iters = len(train_loader) * epochs
    global_step = 0

    logger.info("=" * 60)
    logger.info("Training Configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {cfg['hyperparams']['batch_size']}")
    logger.info(f"  Learning rate: {cfg['hyperparams']['lr']}")
    logger.info(f"  Total iterations: {total_iters}")
    logger.info(f"  Validation interval: {val_interval} iters")
    logger.info(f"  QDTrack pos_iou_thr: {pos_iou_thr}")
    logger.info(f"  QDTrack neg_iou_thr: {neg_iou_thr}")
    logger.info(f"  Samples (key/ref): {num_samples_key}/{num_samples_ref}")
    logger.info("=" * 60)

    # Initial GPU memory check
    log_gpu_memory(logger)

    logger.info("===== TRAINING START =====")

    # ---------------------------
    # Training Loop
    # ---------------------------
    start_time = time.time()

    for epoch in range(epochs):
        logger.info("=" * 60)
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info("=" * 60)

        model.train()
        epoch_losses = {
            'loss_total': 0.0,
            'loss_rpn': 0.0,
            'loss_cls': 0.0,
            'loss_reg': 0.0,
            'loss_contrastive': 0.0,
            'loss_aux': 0.0,
            'loss_reid': 0.0
        }

        for iter_i, batch in enumerate(train_loader):
            global_step += 1

            # Unpack temporal pair batch
            (frames_key, frames_ref,
             boxes_key, boxes_ref,
             ids_key, ids_ref,
             metas) = batch

            # Move to device
            frames_key = frames_key.to(device, non_blocking=True)
            frames_ref = frames_ref.to(device, non_blocking=True)
            boxes_key = [b.to(device, non_blocking=True) for b in boxes_key]
            boxes_ref = [b.to(device, non_blocking=True) for b in boxes_ref]
            ids_key = [i.to(device, non_blocking=True) for i in ids_key]
            ids_ref = [i.to(device, non_blocking=True) for i in ids_ref]

            optimizer.zero_grad()

            # Forward with AMP
            if use_amp:
                with autocast():
                    losses = model.forward_train(
                        images_key=frames_key,
                        images_ref=frames_ref,
                        gt_boxes_key=boxes_key,
                        gt_boxes_ref=boxes_ref,
                        gt_ids_key=ids_key,
                        gt_ids_ref=ids_ref,
                        pos_iou_thr=pos_iou_thr,
                        neg_iou_thr=neg_iou_thr,
                        num_samples_key=num_samples_key,
                        num_samples_ref=num_samples_ref
                    )
                    loss_total = losses["loss_total"]

                # Backward with gradient scaling
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward without AMP
                losses = model.forward_train(
                    images_key=frames_key,
                    images_ref=frames_ref,
                    gt_boxes_key=boxes_key,
                    gt_boxes_ref=boxes_ref,
                    gt_ids_key=ids_key,
                    gt_ids_ref=ids_ref,
                    pos_iou_thr=pos_iou_thr,
                    neg_iou_thr=neg_iou_thr,
                    num_samples_key=num_samples_key,
                    num_samples_ref=num_samples_ref
                )
                loss_total = losses["loss_total"]

                # Backward
                loss_total.backward()
                optimizer.step()

            # Accumulate losses for epoch average
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()

            # Logging
            if global_step == 1 or global_step % log_interval == 0:
                elapsed = time.time() - start_time
                iters_done = global_step
                iters_left = total_iters - global_step
                eta_seconds = (elapsed / iters_done) * iters_left if iters_done > 0 else 0

                eta_h = int(eta_seconds // 3600)
                eta_m = int((eta_seconds % 3600) // 60)

                logger.info(
                    f"[Iter {iter_i + 1}/{len(train_loader)}] "
                    f"[Global {global_step}/{total_iters}] "
                    f"[Epoch {epoch + 1}/{epochs}] "
                    f"Loss={loss_total.item():.4f} "
                    f"rpn={losses['loss_rpn'].item():.4f} "
                    f"cls={losses['loss_cls'].item():.4f} "
                    f"reg={losses['loss_reg'].item():.4f} "
                    f"cont={losses['loss_contrastive'].item():.4f} "
                    f"aux={losses['loss_aux'].item():.4f} "
                    f"ETA={eta_h}h{eta_m}m"
                )

            # Validation & Checkpointing
            if global_step % val_interval == 0:
                logger.info("=" * 60)
                logger.info(f"Validation @ Iteration {global_step}")
                logger.info("=" * 60)

                # Save iteration checkpoint
                ckpt_manager.save_iter(model, optimizer, global_step, epoch)

                # Log GPU memory before validation
                log_gpu_memory(logger)

                # Validation (if available)
                model.eval()
                val_metrics = validate_qdtrack(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    logger=logger,
                    max_batches=cfg["hyperparams"].get("val_max_batches", None)  # Limit for speed
                )

                # === Save per-camera AVI with iteration number ===
                for seq_id in set([r["scenario"] for r in val_metrics["detection_results"]]):
                    seq_results = [r for r in val_metrics["detection_results"] if r["scenario"] == seq_id]

                    save_path = os.path.join(save_root, f"val_step{global_step}_{seq_id}.avi")
                    save_visdrone_video(seq_results, cfg["val"]["video_root"], save_path)
                    logger.info(f"[VAL VIDEO SAVED] {save_path}")


                # Compute detection metrics
                det_metrics = compute_detection_metrics(
                    val_metrics['detection_results'],
                    iou_threshold=0.5
                )

                logger.info("Detection Metrics:")
                logger.info(f"  Precision: {det_metrics['precision']:.4f}")
                logger.info(f"  Recall: {det_metrics['recall']:.4f}")
                logger.info(f"  F1-Score: {det_metrics['f1']:.4f}")
                logger.info(f"  TP: {det_metrics['tp']}, FP: {det_metrics['fp']}, FN: {det_metrics['fn']}")

                # Save best model based on F1 score
                current_f1 = det_metrics['f1']
                if not hasattr(ckpt_manager, 'best_f1') or current_f1 > ckpt_manager.best_f1:
                    ckpt_manager.best_f1 = current_f1
                    best_path = os.path.join(save_root, "best_model.pth")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'f1_score': current_f1,
                        'metrics': det_metrics
                    }, best_path)
                    logger.info(f"New best model saved! F1: {current_f1:.4f}")

                model.train()
                logger.info("=" * 60)

        # End of epoch
        train_dataset.clear_cache()
        cache_stats = train_dataset.get_cache_stats()
        if cache_stats:
            logger.info(
                f"Frame cache stats: "
                f"hits={cache_stats['hits']}, "
                f"misses={cache_stats['misses']}, "
                f"hit_rate={cache_stats['hit_rate']:.2%}, "
                f"size={cache_stats['size']}"
            )

        clear_gpu_cache(logger)

        # Epoch statistics
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)

        logger.info("=" * 60)
        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Total Loss: {epoch_losses['loss_total']:.4f}")
        logger.info(f"  RPN Loss: {epoch_losses['loss_rpn']:.4f}")
        logger.info(f"  Cls Loss: {epoch_losses['loss_cls']:.4f}")
        logger.info(f"  Reg Loss: {epoch_losses['loss_reg']:.4f}")
        logger.info(f"  Contrastive Loss: {epoch_losses['loss_contrastive']:.4f}")
        logger.info(f"  Aux Loss: {epoch_losses['loss_aux']:.4f}")
        logger.info(f"  ReID Loss: {epoch_losses['loss_reid']:.4f}")

        # Learning rate info
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        logger.info("=" * 60)

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Log GPU memory
        log_gpu_memory(logger)

    # ---------------------------
    # Training Complete
    # ---------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    total_time = time.time() - start_time
    total_h = int(total_time // 3600)
    total_m = int((total_time % 3600) // 60)

    logger.info(f"Total training time: {total_h}h {total_m}m")
    logger.info(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Save final model
    final_path = os.path.join(save_root, "final_model.pth")
    torch.save({
        'epoch': epochs,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    logger.info(f"Final model saved: {final_path}")

    # Final GPU memory check
    log_gpu_memory(logger)

    # Cleanup
    train_dataset.video_pool.release_all()
    logger.info("Video resources released")


if __name__ == "__main__":
    train()