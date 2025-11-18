import os
import yaml
import torch
import time
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.transforms as T
from tqdm import tqdm

from datasets.dataloader import data_loader
from inference.single_inference import SingleCameraInference
from inference.multi_inference import MultiCameraInference
from inference.export_utils import save_mot_txt, save_coco_json, save_single_camera_avi, merge_multi_camera_avi

from utils.logger import setup_logger
from utils.utils_print import print_val_summary, save_tracking_csv
from utils.checkpoint import CheckpointManager

from models.detector import DetectorWithReID
from models.res50_backbone import FPNBackbone
from models.rpn_head import RPNHead
from models.roi import RoIAlignLayer
from models.bbox_head import BBoxHead
from models.reid_head import ReIDHead

from tracking.single_camera import SingleCameraTracker
from tracking.eval import validate_tracking
from tracking.validation_multi import validate_multi_camera

# -----------------------------------------------------------
# Build model
# -----------------------------------------------------------
def build_detector_with_reid(device, num_ids):
    backbone = FPNBackbone()
    rpn_head = RPNHead(
        in_channels=256,
        num_anchors=9,
        strides=[8,16,32,64],
        scales=[4,8,16],
        ratios=[0.5,1.0,2.0],
        pre_nms_topk=1000,
        post_nms_topk=200,
        nms_thresh=0.7
    )
    roi_align = RoIAlignLayer(output_size=7)
    bbox_head = BBoxHead(in_channels=256, fc_dim=1024, num_classes=1, score_thresh=0.0)
    reid_head = ReIDHead(in_channels=256, embed_dim=256, num_ids=num_ids)
    model = DetectorWithReID(backbone, rpn_head, roi_align, bbox_head, reid_head)
    return model.to(device)


# -----------------------------------------------------------
# Train
# -----------------------------------------------------------
def train():

    # ---------------------------
    # Load Config
    # ---------------------------
    with open("./pknu_mtmdc.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup logger
    timestamp = str(int(time.time() * 1000))
    save_root = os.path.join(cfg["log"]["save_dir"], timestamp)
    os.makedirs(save_root, exist_ok=True)
    log_file = os.path.join(save_root, "train.log")
    logger = setup_logger(log_file)
    ckpt_manager = CheckpointManager(save_root, logger)

    logger.info("===== TRAINING START =====")

    # ---------------------------
    # Transform
    # ---------------------------
    transform = T.Compose([
        # T.ToPILImage(),
        # T.Resize((540, 960)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # ---------------------------
    # Load dataset
    # ---------------------------
    train_loader = data_loader(
        cfg["train"]["video_root"],
        cfg["train"]["ann_root"],
        cfg["scenario"]["train_ids"][:],
        transform,
        batch_size=cfg["hyperparams"]["batch_size"],
        frame_stride=cfg["hyperparams"]["frame_stride"]
    )

    val_loader = data_loader(
        cfg["val"]["video_root"],
        cfg["val"]["ann_root"],
        cfg["scenario"]["val_ids"][:],
        transform,
        batch_size=1,
        frame_stride=1
    )

    # ---------------------------
    # Model & Optimizer
    # ---------------------------
    model = build_detector_with_reid(device, cfg["model"]["num_ids"])
    optimizer = Adam(model.parameters(), lr=cfg["hyperparams"]["lr"])

    epochs = cfg["hyperparams"]["epochs"]
    VAL_INTERVAL = cfg["hyperparams"]["val_interval"]
    warmup_epochs = cfg["hyperparams"]["warmup_epochs"]

    total_iters = len(train_loader) * epochs

    logger.info(f"Total train samples: {len(train_loader.dataset)}")
    logger.info(f"Total iters per epoch: {len(train_loader)}")
    logger.info(f"Total iters: {total_iters}")

    global_step = 0

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(epochs):
        logger.info(f"===== Epoch {epoch+1}/{epochs} =====")

        start_time = time.time()
        for iter_i, (frames, boxes, labels, tids, pids, metas) in enumerate(train_loader):
            global_step += 1
            frames = frames.to(device)

            gt_boxes  = [b.to(device) for b in boxes]
            gt_labels = [l.to(device) for l in labels]
            gt_ids    = [p.to(device) for p in pids]

            optimizer.zero_grad()

            losses = model.forward_train(
                frames, gt_boxes, gt_labels, gt_ids,
                epoch=epoch,
                warmup_epochs=warmup_epochs
            )

            loss_total = losses["loss_total"]
            loss_total.backward()
            optimizer.step()

            # ETA 계산
            elapsed = time.time() - start_time
            remain = (total_iters - global_step) * (elapsed / global_step)

            eta_h = int(remain // 3600)
            eta_m = int((remain % 3600) // 60)

            if global_step == 1 or global_step % 40 == 0:
                logger.info(
                    f"[Epoch {epoch + 1}/{epochs}] "
                    f"[Iter {iter_i + 1}/{len(train_loader)} | Global {global_step}/{total_iters}] "
                    f"Loss={loss_total.item():.4f} "
                    f"cls={losses['loss_cls'].item():.4f} "
                    f"reg={losses['loss_reg'].item():.4f} "
                    f"reid={losses['loss_reid'].item():.4f} "
                    f"ETA={eta_h}h{eta_m}m"
                )

            # Iteration Validation
            if global_step % VAL_INTERVAL == 0:
                logger.info(f"[Validation] iter={global_step}")
                ckpt_manager.save_iter(model, optimizer, global_step, epoch)

                ########################################
                # 1) SINGLE CAMERA VALIDATION
                ########################################

                summary_single = validate_tracking(model, val_loader, SingleCameraTracker, device)
                print_val_summary(summary_single, title=f"SINGLE@Iter {global_step}")

                val_engine_single = SingleCameraInference(model, device)
                val_single = val_engine_single.run(val_loader)

                # SINGLE 저장 디렉토리
                single_dir = os.path.join(save_root, "single", str(global_step))
                os.makedirs(single_dir, exist_ok=True)

                # 저장
                save_tracking_csv(val_single, os.path.join(single_dir, "tracking.csv"))
                save_mot_txt(val_single, os.path.join(single_dir, "mot.txt"))
                save_coco_json(val_single, os.path.join(single_dir, "coco.json"))

                # 단일 카메라 AVI 저장
                cam_ids = sorted(list(set([r["cam_id"] for r in val_single])))

                for cam in cam_ids:
                    cam_results = [r for r in val_single if r["cam_id"] == cam]
                    save_single_camera_avi(
                        results=cam_results,
                        video_root=cfg["val"]["video_root"],
                        save_path=os.path.join(single_dir, f"{cam}.avi")
                    )

                ########################################
                # 2) MULTI-CAMERA (MCTA) VALIDATION
                ########################################

                summary_multi = validate_multi_camera(model, val_loader, device)
                print_val_summary(summary_multi, title=f"MULTI(MCTA)@Iter {global_step}")

                val_engine_multi = MultiCameraInference(model, device)
                val_global = val_engine_multi.run(val_loader)

                # MCTA 저장 디렉토리
                mcta_dir = os.path.join(save_root, "mcta", str(global_step))
                os.makedirs(mcta_dir, exist_ok=True)

                # 저장
                save_tracking_csv(val_global, os.path.join(mcta_dir, "tracking.csv"))
                save_mot_txt(val_global, os.path.join(mcta_dir, "mot.txt"))
                save_coco_json(val_global, os.path.join(mcta_dir, "coco.json"))
                merge_multi_camera_avi(
                    val_global,
                    cfg["val"]["video_root"],
                    os.path.join(mcta_dir, "merged.avi")
                )

                try:
                    cur_single_idf1 = float(summary_single.loc["acc", "idf1"])
                except Exception:
                    cur_single_idf1 = -1.0

                ckpt_manager.save_best_single(
                    model=model,
                    optimizer=optimizer,
                    global_step=global_step,
                    epoch=epoch,
                    cur_idf1=cur_single_idf1
                )

                # MCTA IDF1
                try:
                    cur_mcta_idf1 = float(summary_multi.loc["acc", "idf1"])
                except Exception:
                    cur_mcta_idf1 = -1.0

                ckpt_manager.save_best_mcta(
                    model=model,
                    optimizer=optimizer,
                    global_step=global_step,
                    epoch=epoch,
                    cur_idf1=cur_mcta_idf1
                )

        logger.info(f"Epoch {epoch+1} finished.")

    logger.info("===== TRAINING COMPLETE =====")


if __name__ == "__main__":
    train()