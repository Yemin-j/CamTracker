import os
import yaml
import torch
import time
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.transforms as T
from tqdm import tqdm

from datasets.dataloader import data_loader
from inference.dataloader_builder import test_data_loader
# from inference.single_inference import SingleCameraInference
# from inference.multi_inference import MultiCameraInference

from utils.export_utils import save_mot_txt, save_coco_json, save_single_camera_avi, merge_multi_camera_avi
from utils.logger import setup_logger
from utils.utils_print import print_val_summary, save_tracking_csv
from utils.checkpoint import CheckpointManager

from models.detector import DetectorWithReID
from models.res50_backbone import FPNBackbone
from models.rpn_head import RPNHead
from models.roi import RoIAlignLayer
from models.bbox_head import BBoxHead
from models.reid_head import ReIDHead

# from tracking.single_camera import SingleCameraTracker
# from tracking.eval import validate_tracking
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
    timestamp = 'train_' + timestamp
    save_root = os.path.join(cfg["log"]["save_dir"], timestamp)
    os.makedirs(save_root, exist_ok=True)
    log_file = os.path.join(save_root, "train.log")
    logger = setup_logger(log_file)
    ckpt_manager = CheckpointManager(save_root, logger)

    # ---------------------------
    # Transform
    # ---------------------------
    transform = T.Compose([
        T.Resize((540, 960)),
        T.ToTensor(),
        # T.Normalize([103.530, 116.280, 123.675],[1.0, 1.0, 1.0])
    ])

    # ---------------------------
    # Load dataset
    # ---------------------------
    train_loader = data_loader(
        cfg["train"]["video_root"],
        cfg["train"]["ann_root"],
        cfg["scenario"]["train_ids"][:],
        transform=None,
        batch_size=cfg["hyperparams"]["batch_size"],
        frame_stride=cfg["hyperparams"]["frame_stride"]
    )

    val_loader = test_data_loader(
        cfg["val"]["video_root"],
        cfg["val"]["ann_root"],
        cfg["scenario"]["val_ids"][:],
        transform=None,
        batch_size=1,
        frame_stride=1
    )

    # ---------------------------
    # Model & Optimizer
    # ---------------------------
    model = build_detector_with_reid(device, cfg["model"]["num_ids"])
    optimizer = Adam(model.parameters(), lr=cfg["hyperparams"]["lr"])

    epochs = cfg["hyperparams"]["epochs"]
    val_interval = cfg["hyperparams"]["val_interval"]
    warmup_iter = cfg["hyperparams"]["warmup_iters"]

    total_iters = len(train_loader) * epochs

    logger.info(f"Total train samples: {len(train_loader.dataset)}")
    logger.info(f"Total iters per epoch: {len(train_loader)}")
    logger.info(f"Total iters: {total_iters}")

    global_step = 0

    logger.info("===== TRAINING START =====")

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
            gt_ids    = [p.to(device) for p in pids]

            optimizer.zero_grad()

            losses = model.forward_train(
                frames, gt_boxes, gt_ids,
                epoch=epoch,
                global_step=global_step,
                warmup_iters=warmup_iter
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
                    f"[Iter {iter_i + 1}/{len(train_loader)} | Global {global_step}/{total_iters}] "
                    f"[Epoch {epoch + 1}/{epochs}] "
                    f"Loss={loss_total.item():.4f} "
                    f"cls={losses['loss_cls'].item():.4f} "
                    f"reg={losses['loss_reg'].item():.4f} "
                    f"reid={losses['loss_reid'].item():.4f} "
                    f"ETA={eta_h}h{eta_m}m"
                )

            # Iteration Validation
            if global_step % val_interval == 0:
                logger.info(f"[Validation] iter={global_step}")
                ckpt_manager.save_iter(model, optimizer, global_step, epoch)

                ########################################
                # VALIDATION (Single + MCTA, 1-pass)
                ########################################
                summary_single, summary_mcta, single_results, global_results = \
                    validate_multi_camera(model, val_loader, device, logger=logger,
                                          export_per_cam=True, video_root=cfg["val"]["video_root"]) # export_per_cam은 주로 False

                # 콘솔 출력
                print_val_summary(summary_single, title=f"SINGLE@Iter {global_step}")
                print_val_summary(summary_mcta, title=f"MCTA@Iter {global_step}")

                ########################################
                # 결과 저장
                ########################################

                # SINGLE export
                single_dir = os.path.join(save_root, "single", str(global_step))
                os.makedirs(single_dir, exist_ok=True)

                save_tracking_csv(single_results, os.path.join(single_dir, "tracking.csv"))
                save_mot_txt(single_results, os.path.join(single_dir, "mot.txt"))
                save_coco_json(single_results, os.path.join(single_dir, "coco.json"))

                cam_ids = sorted(list(set(r["cam_id"] for r in single_results)))
                for cam in cam_ids:
                    cam_res = [r for r in single_results if r["cam_id"] == cam]
                    save_single_camera_avi(
                        results=cam_res,
                        video_root=cfg["val"]["video_root"],
                        save_path=os.path.join(single_dir, f"{cam}.avi")
                    )

                # MCTA export
                mcta_dir = os.path.join(save_root, "mcta", str(global_step))
                os.makedirs(mcta_dir, exist_ok=True)

                save_tracking_csv(global_results, os.path.join(mcta_dir, "tracking.csv"))
                save_mot_txt(global_results, os.path.join(mcta_dir, "mot.txt"))
                save_coco_json(global_results, os.path.join(mcta_dir, "coco.json"))
                merge_multi_camera_avi(
                    global_results,
                    cfg["val"]["video_root"],
                    os.path.join(mcta_dir, "merged.avi")
                )

                ########################################
                # BEST checkpoint (single + mcta 평균)
                ########################################
                try:
                    idf1_single = float(summary_single.loc['acc', 'idf1'])
                except Exception:
                    idf1_single = 0.0

                try:
                    idf1_mcta = float(summary_mcta.loc['acc', 'idf1'])
                except Exception:
                    idf1_mcta = 0.0

                combined_score = (idf1_single + idf1_mcta) / 2.0

                ckpt_manager.save_best(
                    model=model,
                    optimizer=optimizer,
                    global_step=global_step,
                    epoch=epoch,
                    combined_score=combined_score
                )

        logger.info(f"Epoch {epoch+1} finished.")

    logger.info("===== TRAINING COMPLETE =====")


if __name__ == "__main__":
    train()