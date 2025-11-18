import os
import torch

class CheckpointManager:
    def __init__(self, save_root, logger=None):
        """
        save_root: 기본 결과 디렉토리 (ex: D:/tar_trac/results)
        logger: logging.Logger (optional)
        """
        self.save_root = save_root
        self.logger = logger

        # best metric 저장용
        self.best_single_idf1 = -1.0
        self.best_mcta_idf1 = -1.0

        # 체크포인트 저장 디렉토리
        self.ckpt_dir = os.path.join(self.save_root, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def save_iter(self, model, optimizer, global_step, epoch):
        """
        iteration 기준 일반 체크포인트 저장
        """
        ckpt_path = os.path.join(self.ckpt_dir, f"iter_{global_step}.pt")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, ckpt_path)
        self._log(f"[CKPT] Saved iter checkpoint @step={global_step} → {ckpt_path}")

    def save_best_single(self, model, optimizer, global_step, epoch, cur_idf1):
        """
        single-camera IDF1 기준 best 모델 저장
        """
        if cur_idf1 <= self.best_single_idf1:
            return  # 개선 안 됨

        self.best_single_idf1 = cur_idf1

        ckpt_path = os.path.join(self.ckpt_dir, "best_single.pt")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "best_single_idf1": self.best_single_idf1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, ckpt_path)
        self._log(f"[BEST] Updated BEST SINGLE model @step={global_step}, IDF1={cur_idf1:.4f}")

    def save_best_mcta(self, model, optimizer, global_step, epoch, cur_idf1):
        """
        multi-camera(MCTA) IDF1 기준 best 모델 저장
        """
        if cur_idf1 <= self.best_mcta_idf1:
            return  # 개선 안 됨

        self.best_mcta_idf1 = cur_idf1

        ckpt_path = os.path.join(self.ckpt_dir, "best_mcta.pt")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "best_mcta_idf1": self.best_mcta_idf1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, ckpt_path)
        self._log(f"[BEST] Updated BEST MCTA model @step={global_step}, IDF1={cur_idf1:.4f}")
