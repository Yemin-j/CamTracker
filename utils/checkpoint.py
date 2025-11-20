import os
import torch

class CheckpointManager:
    def __init__(self, save_root, logger=None):
        self.save_root = save_root
        self.logger = logger
        self.best_score = -1.0

        self.ckpt_dir = os.path.join(save_root, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def save_iter(self, model, optimizer, global_step, epoch):
        ckpt_path = os.path.join(self.ckpt_dir, f"iter_{global_step}.pth")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, ckpt_path)
        self._log(f"[CKPT] Saved iteration checkpoint: {ckpt_path}")

    def save_best(self, model, optimizer, global_step, epoch, combined_score):
        if combined_score <= self.best_score:
            return

        self.best_score = combined_score

        ckpt_path = os.path.join(self.ckpt_dir, f"best_iter_{global_step}.pth")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "best_score": combined_score,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, ckpt_path)

        self._log(f"[BEST] Updated BEST checkpoint @iter={global_step}, score={combined_score:.4f}")