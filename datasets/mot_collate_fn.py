import torch

class custom_collate_fn:
    def __call__(self, batch):
        frames = []
        boxes = []
        labels = []
        tids = []
        pids = []
        metas = []

        for item in batch:
            frame, box, lab, tid, pid, meta = item
            frames.append(frame)  # (3,H,W)
            boxes.append(box)  # (N,4)
            labels.append(lab)  # (N,)
            tids.append(tid)
            pids.append(pid)
            metas.append(meta)

        frames = torch.stack(frames, dim=0)  # (B,3,H,W)

        # 나머지는 list 그대로 (variable length라서 stack 불가능)
        return frames, boxes, labels, tids, pids, metas