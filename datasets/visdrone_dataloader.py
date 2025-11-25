import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random


class VideoReaderPool:
    def __init__(self):
        self.pool = {}

    def get(self, path):
        if path not in self.pool:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video {path}")
            self.pool[path] = cap
        return self.pool[path]

    def release_all(self):
        for cap in self.pool.values():
            cap.release()
        self.pool.clear()


class VisDroneQDTrackDataset(Dataset):
    """
    VisDrone 2019-MOT -> QDTrack 학습용 temporal pair dataset
    """

    def __init__(self,
                 video_root,
                 ann_root,
                 scenario_ids,
                 frame_stride=1,
                 temporal_offset_range=3,
                 resize_w=960,
                 resize_h=540,
                 transform=None):

        self.video_root = video_root
        self.ann_root = ann_root
        self.scenario_ids = scenario_ids
        self.frame_stride = frame_stride
        self.temporal_offset_range = temporal_offset_range
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.transform = transform

        self.video_pool = VideoReaderPool()

        # Build sequences & temporal pairs
        self.sequences, self.pairs = self._build_index()

    def _build_index(self):
        sequences = {}
        pairs = []

        for seq in self.scenario_ids:
            # seq = seq + '/'
            img_dir = os.path.join(self.video_root, seq)
            ann_path = os.path.join(self.ann_root, seq + ".txt")
            if not os.path.isdir(img_dir) or not os.path.exists(ann_path):
                continue

            # Load annotation by frame
            gt_by_frame = defaultdict(list)
            with open(ann_path, "r") as f:
                for line in f:
                    fr, tid, x, y, w, h, score, cat, trunc, occ = map(int, line.split(","))

                    if tid <= 0:  # ignore id 0
                        continue

                    gt_by_frame[fr].append(
                        {
                            "track_id": tid,
                            "bbox": [x, y, x + w, y + h]  # xyxy 형태
                        }
                    )

            # Frame ids
            images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
            frame_ids = sorted([int(f.split(".")[0]) for f in images])

            # Apply frame stride
            frame_ids = [fid for fid in frame_ids if fid % self.frame_stride == 0]
            if len(frame_ids) < 2:
                continue

            sequences[seq] = {
                "img_dir": img_dir,
                "ann_path": ann_path,
                "frame_ids": frame_ids,
                "gt_by_frame": gt_by_frame
            }

            # Build valid pairs
            for i, kfid in enumerate(frame_ids):
                key_ids = set([obj["track_id"] for obj in gt_by_frame.get(kfid, [])])
                if len(key_ids) == 0:
                    continue

                start = max(0, i - self.temporal_offset_range)
                end = min(len(frame_ids), i + self.temporal_offset_range + 1)

                for j in range(start, end):
                    if i == j:
                        continue
                    rfid = frame_ids[j]

                    ref_ids = set([obj["track_id"] for obj in gt_by_frame.get(rfid, [])])

                    if key_ids.intersection(ref_ids):
                        pairs.append({
                            "seq": seq,
                            "key": kfid,
                            "ref": rfid
                        })

        return sequences, pairs

    def __len__(self):
        return len(self.pairs)

    def _load_frame(self, img_path):
        """
        VisDrone에서 한 프레임(.jpg)을 로드하고 960x540으로 리사이즈해서 텐서로 반환
        """
        # 그냥 이미지 로딩 (비디오 캡쳐 X)
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # 리사이즈
        img = cv2.resize(img, (self.resize_w, self.resize_h))

        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return tensor, (orig_w, orig_h)

    def _load_ann(self, ann_path, frame_id, orig_w, orig_h):
        boxes, ids = [], []

        scale_x = self.resize_w / orig_w
        scale_y = self.resize_h / orig_h

        with open(ann_path, "r") as f:
            for line in f:
                fr, tid, x, y, w, h, score, cat, trunc, occ = map(int, line.split(","))

                if fr != frame_id:
                    continue
                if tid <= 0:
                    continue
                if cat not in [1, 2]: # train only person
                    continue

                x1 = x * scale_x
                y1 = y * scale_y
                x2 = (x + w) * scale_x
                y2 = (y + h) * scale_y

                boxes.append([x1, y1, x2, y2])
                ids.append(tid)

        if len(boxes) == 0:
            return torch.zeros((0, 4)), torch.zeros((0,))

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        seq = item["seq"]
        kfid = item["key"]
        rfid = item["ref"]

        seq_info = self.sequences[seq]

        key_img = os.path.join(seq_info["img_dir"], f"{kfid:07d}.jpg")
        ref_img = os.path.join(seq_info["img_dir"], f"{rfid:07d}.jpg")

        # Load frames
        frame_key, (orig_w_k, orig_h_k) = self._load_frame(key_img)
        frame_ref, (orig_w_r, orig_h_r) = self._load_frame(ref_img)

        # Load GT
        boxes_key, ids_key = self._load_ann(seq_info["ann_path"], kfid, orig_w_k, orig_h_k)
        boxes_ref, ids_ref = self._load_ann(seq_info["ann_path"], rfid, orig_w_r, orig_h_r)

        if self.transform:
            frame_key = self.transform(frame_key)
            frame_ref = self.transform(frame_ref)

        meta = {
            "seq": seq,
            "key_frame": kfid,
            "ref_frame": rfid
        }

        return frame_key, frame_ref, boxes_key, boxes_ref, ids_key, ids_ref, meta

class VisDroneSingleFrameDataset(Dataset):
    """
    VisDrone 2019-MOT Single Frame dataset.
    - training이 아닌 validation/test/tracking inference용
    - temporal pair가 필요 없어서 frame drop 없음 (GT 없는 frame도 허용)
    - tracking inference(t→t-1)에서도 자연스럽게 사용 가능
    """

    def __init__(self,
                 video_root,
                 ann_root,
                 scenario_ids,
                 resize_w=960,
                 resize_h=540,
                 frame_stride=1,
                 transform=None):
        self.video_root = video_root
        self.ann_root = ann_root
        self.scenario_ids = scenario_ids
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.frame_stride = frame_stride
        self.transform = transform

        self.items = self._build_index()

    def _build_index(self):
        items = []  # (seq_id, frame_id, img_path, ann_path)

        for seq in self.scenario_ids:
            img_dir = os.path.join(self.video_root, seq)
            ann_path = os.path.join(self.ann_root, seq + ".txt")

            if not os.path.isdir(img_dir) or not os.path.exists(ann_path):
                continue

            # 모든 프레임 로드
            imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
            # imgs = sorted(imgs, key=lambda f: int(f.replace('.jpg','')))

            for f in imgs:
                frame_id = int(f.replace(".jpg", ""))

                # stride 적용
                if frame_id % self.frame_stride != 0:
                    continue

                items.append({
                    "seq": seq,
                    "frame_id": frame_id,
                    "img_path": os.path.join(img_dir, f),
                    "ann_path": ann_path,
                })

        return items

    def __len__(self):
        return len(self.items)

    # frame loader 동일
    def _load_frame(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (self.resize_w, self.resize_h))
        tensor = torch.from_numpy(img).permute(2,0,1).float()/255.0
        return tensor, (orig_w, orig_h)

    # GT loader 동일 + person-only filter
    def _load_ann(self, ann_path, frame_id, orig_w, orig_h):
        boxes, ids = [], []
        scale_x = self.resize_w / orig_w
        scale_y = self.resize_h / orig_h

        with open(ann_path,"r") as f:
            for line in f:
                fr, tid, x, y, w, h, score, cat, trunc, occ = map(int, line.split(","))
                if fr != frame_id:
                    continue
                if tid <= 0:
                    continue
                if cat not in [1, 2]:   # person only
                    continue

                x1 = x * scale_x
                y1 = y * scale_y
                x2 = (x + w) * scale_x
                y2 = (y + h) * scale_y
                boxes.append([x1, y1, x2, y2])
                ids.append(tid)

        if len(boxes) == 0:
            return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.long)
        return torch.tensor(boxes,dtype=torch.float32), torch.tensor(ids,dtype=torch.long)

    def __getitem__(self, idx):
        item = self.items[idx]
        seq = item["seq"]
        frame_id = item["frame_id"]
        img_path = item["img_path"]
        ann_path = item["ann_path"]

        frame, (orig_w, orig_h) = self._load_frame(img_path)
        boxes, ids = self._load_ann(ann_path, frame_id, orig_w, orig_h)

        if self.transform:
            frame = self.transform(frame)

        meta = {
            "seq": seq,
            "frame_id": frame_id,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "resize_w": self.resize_w,
            "resize_h": self.resize_h
        }

        # validation_qdtrack 호환 포맷:
        # (frames, boxes, labels, tids, pids, metas)
        return frame, boxes, None, ids, None, meta

