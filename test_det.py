import os
import time
import yaml
import torch
import argparse

from utils.logger import setup_logger
from inference.validation_qdtrack import (
    build_validation_dataloader,
    validate_qdtrack,
    compute_detection_metrics,
)
from utils.export_utils import save_visdrone_video
from train import build_qdtrack_model  # train.py에 이미 정의되어 있음


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="pknu_mtmdc.yaml")
    p.add_argument("--ckpt", type=str, default='D:/tar_trac/result/qdtrack_train_1763962526270/checkpoints/iter_19200.pth',
                   help="직접 체크포인트 경로 지정 (없으면 best_model.pth 사용)")
    return p.parse_args()


def main():
    args = parse_args()

    # --------------------
    # 1) config / device
    # --------------------
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------
    # 2) 로그 세팅
    # --------------------
    timestamp = "test_det_" + str(int(time.time() * 1000))
    save_root = os.path.join(cfg["log"]["save_dir"], timestamp)
    os.makedirs(save_root, exist_ok=True)
    log_file = os.path.join(save_root, "test_det.log")
    logger = setup_logger(log_file)
    logger.info(f"[TEST-DET] save_root = {save_root}")

    # --------------------
    # 3) 모델 빌드 (train 때와 동일)
    # --------------------
    model = build_qdtrack_model(cfg, device=device)  #

    # --------------------
    # 4) 체크포인트 로드
    # --------------------
    if args.ckpt is not None:
        ckpt_path = args.ckpt
    else:
        # train.py에서 best_model.pth 로 저장했으니까 그걸 기본값으로 사용
        ckpt_path = os.path.join(cfg["log"]["save_dir"],
                                 cfg["test"].get("ckpt_exp", ""),
                                 "best_model.pth")

    logger.info(f"[TEST-DET] Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state:
        # 정상적으로 저장된 경우
        model.load_state_dict(state["model_state_dict"])
    elif "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        # 순수 state_dict로 저장된 iteration checkpoints
        model.load_state_dict(state)
    model.to(device)

    # --------------------
    # 5) validation/test dataloader 생성
    #    (build_validation_dataloader 는 우리가 재작성한 temporal용 함수)
    # --------------------
    val_loader = build_validation_dataloader(cfg, device)  #
    logger.info(f"[TEST-DET] Validation dataset size: {len(val_loader.dataset)}")

    # --------------------
    # 6) validate_qdtrack 으로 detection-only 평가
    # --------------------
    metrics = validate_qdtrack(model, val_loader, device, logger=logger)  #

    det_metrics = compute_detection_metrics(metrics["detection_results"])
    logger.info("[TEST-DET] Detection Metrics:")
    logger.info(f"  Precision: {det_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {det_metrics['recall']:.4f}")
    logger.info(f"  F1-score:  {det_metrics['f1']:.4f}")
    logger.info(f"  TP: {det_metrics['tp']}, FP: {det_metrics['fp']}, FN: {det_metrics['fn']}")

    print("\n===== TEST DET RESULT =====")
    print(f"Precision: {det_metrics['precision']:.4f}")
    print(f"Recall:    {det_metrics['recall']:.4f}")
    print(f"F1-score:  {det_metrics['f1']:.4f}")
    print(f"TP: {det_metrics['tp']}, FP: {det_metrics['fp']}, FN: {det_metrics['fn']}")
    print(f"Log saved to: {log_file}")
    print("===========================\n")

    img_root = cfg["val"]["video_root"]  # ex) sequences/
    for seq_id in set([r["scenario"] for r in metrics["detection_results"]]):
        seq_results = [r for r in metrics["detection_results"] if r["scenario"] == seq_id]

        save_path = os.path.join(save_root, f"{seq_id}.avi")
        save_visdrone_video(seq_results, img_root, save_path)

    logger.info("Done test_det.")

if __name__ == "__main__":
    main()
