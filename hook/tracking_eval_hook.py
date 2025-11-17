class TrackingEvalHook:
    """
    mmtrack의 EvalHook 느낌으로, 학습 중간에 tracking 평가를 실행하는 Hook.
    """
    def __init__(self, val_loader, tracker_cls, device, interval=1):
        self.val_loader = val_loader
        self.tracker_cls = tracker_cls
        self.device = device
        self.interval = interval

    def after_epoch(self, epoch, model):
        # epoch는 0부터 시작한다고 가정
        if (epoch + 1) % self.interval != 0:
            return
        print(f"\n[Eval] epoch {epoch+1} - running tracking validation...")
        from ..tracking.eval import validate_tracking
        summary = validate_tracking(model, self.val_loader, self.tracker_cls, self.device)
        # summary는 pandas DataFrame, 여기서 주요 metric만 뽑아서 log
        try:
            mota = float(summary.loc['acc', 'mota'])
            idf1 = float(summary.loc['acc', 'idf1'])
        except:
            mota = idf1 = -1

        print(f"[VAL] Epoch={epoch + 1} | MOTA={mota:.4f} | IDF1={idf1:.4f}")
        print("==============================================\n")
