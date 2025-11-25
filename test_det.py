import os
import time
import yaml
import torch
from torch.utils.data import DataLoader
import argparse

from utils.logger import setup_logger
from utils.export_utils import save_tracking_avi

from inference.single_inference import SingleCameraInference

from datasets.visdrone_dataloader import VisDroneSingleFrameDataset
from datasets.mot_collate_fn import QDTrackCollateFn
from datasets.load_ids import list_visdrone_sequences

from train import build_qdtrack_model  # train.py에 이미 정의되어 있음



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="pknu_mtmdc.yaml")
    p.add_argument("--ckpt", type=str, default='D:/tar_trac/result/qdtrack_train_1763969133470/checkpoints/iter_16800.pth',
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

    # --------------------------
    # 5)validation/test dataloader 생성
    # --------------------------
    test_ids = list_visdrone_sequences(cfg["val"]["video_root"])
    dataset = VisDroneSingleFrameDataset(
        video_root=cfg["val"]["video_root"],
        ann_root=cfg["val"]["ann_root"],
        scenario_ids=test_ids,
        frame_stride=1,
        resize_w=960,
        resize_h=540,
        transform=None,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=QDTrackCollateFn().simple_collate,
        pin_memory=True
    )
    logger.info(f"[TEST-DET] Validation dataset size: {len(test_loader.dataset)}")

    # --------------------
    # 6) validate_qdtrack 으로 detection-only 평가
    # --------------------
    infer = SingleCameraInference(model, device, logger=logger)  # :contentReference[oaicite:4]{index=4}
    tracking_results = infer.run(test_loader)

    # --------------------------
    # 6) Save AVI (track_id 포함)
    # --------------------------
    seq_ids = set([r["scenario"] for r in tracking_results])
    img_root = cfg["val"]["video_root"]

    for seq in seq_ids:
        seq_res = [t for t in tracking_results if t["scenario"] == seq]
        save_path = os.path.join(save_root, f"{seq}_track.avi")
        save_tracking_avi(seq_res, img_root, save_path)

    logger.info("TEST tracking completed.")


if __name__ == "__main__":
    main()
