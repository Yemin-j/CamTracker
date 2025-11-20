import os
import time
import yaml
import argparse
import torch
import torchvision.transforms as T

from models.detector import DetectorWithReID
from inference.dataloader_builder import test_data_loader
from tracking.validation_multi import validate_multi_camera
from utils.utils_print import save_tracking_csv
from utils.logger import setup_logger
from utils.export_utils import (
    save_mot_txt,
    save_coco_json,
    save_single_camera_avi,
    merge_multi_camera_avi
)

# ----------------------------
# Parse arguments
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="pknu_mtmdc.yaml",
        help="Path to YAML file"
    )
    return parser.parse_args()


# ----------------------------
# Build transform
# ----------------------------
def build_transform(img_h, img_w, mean, std):
    return T.Compose([
        # T.ToPILImage(),
        T.Resize((img_h, img_w)),
        T.ToTensor(),
        # T.Normalize(mean, std)
    ])


# ----------------------------
# Load checkpoint (best or override)
# ----------------------------
def load_checkpoint(model, ckpt_root, logger=print):

    # override: manually specify .pt
    if ckpt_root is not None:
        logger(f"[LOAD] Using custom checkpoint: {ckpt_root}")
        state = torch.load(ckpt_root, map_location="cpu")
        model.load_state_dict(state["model_state"])
        return

    best_mcta = os.path.join(ckpt_root, "best_mcta.pt")
    best_single = os.path.join(ckpt_root, "best_single.pt")

    if os.path.exists(best_mcta):
        logger(f"[LOAD] best_mcta → {best_mcta}")
        state = torch.load(best_mcta, map_location="cpu")
        model.load_state_dict(state["model_state"])
        return

    if os.path.exists(best_single):
        logger(f"[LOAD] best_single → {best_single}")
        state = torch.load(best_single, map_location="cpu")
        model.load_state_dict(state["model_state"])
        return

    raise FileNotFoundError("No checkpoint found: neither best_mcta.pt nor best_single.pt")


# ----------------------------
# MAIN TEST
# ----------------------------
def main():
    args = parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Timestamp
    timestamp = str(int(time.time() * 1000))
    timestamp = 'test_' + timestamp
    save_root = os.path.join(cfg["log"]["save_dir"], timestamp)
    os.makedirs(save_root, exist_ok=True)
    print(f"\n[TEST] Output directory: {save_root}\n")
    log_file = os.path.join(save_root, "test.log")
    logger = setup_logger(log_file)
    logger.info("[TEST] Logger initialized.")

    # ------------------------
    # Build transform
    # ------------------------
    img_h = cfg["data"]["img_h"]
    img_w = cfg["data"]["img_w"]
    mean  = cfg["data"]["norm_mean"]
    std   = cfg["data"]["norm_std"]

    transform = build_transform(img_h, img_w, mean, std)

    # ------------------------
    # Build test loader
    # ------------------------
    test_loader = test_data_loader(
        video_root=cfg["val"]["video_root"],
        ann_root=cfg["val"]["ann_root"],
        scenarios=cfg["scenario"]["val_ids"],
        transform=None,
        batch_size=1
    )

    # ------------------------
    # Build model
    # ------------------------
    num_ids = cfg["model"]["num_ids"]
    model = DetectorWithReID.build_default(num_ids=num_ids, device=device)

    # ------------------------
    # Load checkpoint
    # ------------------------
    ckpt_root = os.path.join(cfg["log"]["save_dir"], cfg["test"]["ckpt_exp"], cfg["test"]["checkpoints"])
    load_checkpoint(model, ckpt_root)

    # ============================================================
    # 1) SINGLE-CAMERA TEST
    # ============================================================
    print("[TEST] Running SINGLE-CAMERA TRACKING...")
    summary_single, summary_mcta, single_results, global_results = \
        validate_multi_camera(model, test_loader, device, logger=logger)

    single_dir = os.path.join(save_root, "single")
    os.makedirs(single_dir, exist_ok=True)

    save_tracking_csv(single_results, os.path.join(single_dir, "tracking.csv"))
    save_mot_txt(single_results, os.path.join(single_dir, "mot.txt"))
    save_coco_json(single_results, os.path.join(single_dir, "coco.json"))

    for cam in sorted(set(r["cam_id"] for r in single_results)):
        cam_results = [r for r in single_results if r["cam_id"] == cam]
        save_single_camera_avi(cam_results, cfg["val"]["video_root"],
                               os.path.join(single_dir, f"{cam}.avi"))

    # 3) Save multi-camera results
    mcta_dir = os.path.join(save_root, "mcta")
    os.makedirs(mcta_dir, exist_ok=True)

    save_tracking_csv(global_results, os.path.join(mcta_dir, "tracking_global.csv"))
    save_mot_txt(global_results, os.path.join(mcta_dir, "mot_global.txt"))
    save_coco_json(global_results, os.path.join(mcta_dir, "coco_global.json"))
    merge_multi_camera_avi(global_results, cfg["val"]["video_root"],
                           os.path.join(mcta_dir, "merged.avi"))

    print("[TEST] Multi-camera MCTA tracking complete.\n")

    print(f"[TEST] All results saved under: {save_root}\n")


if __name__ == "__main__":
    main()
