import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np

class VideoReaderPool:
    """하나의 비디오 파일당 VideoCapture 객체를 미리 열어서 계속 재사용"""
    def __init__(self):
        self.pool = {}

    def get(self, path):
        if path not in self.pool:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {path}")
            self.pool[path] = cap
        return self.pool[path]

    def release_all(self):
        for cap in self.pool.values():
            cap.release()
        self.pool.clear()

class CachedFramePool:
    """메모리 기반 프레임 캐시 (LRU도 가능하지만 simple dict로 충분)"""
    def __init__(self, max_frames=5000):
        self.cache = {}
        self.limit = max_frames

    def get(self, key):
        return self.cache.get(key, None)

    def put(self, key, frame):
        if len(self.cache) >= self.limit:
            # 너무 크면 절반 제거해서 메모리 회수
            keys = list(self.cache.keys())[: self.limit // 2]
            for k in keys:
                self.cache.pop(k, None)
        self.cache[key] = frame


class AIMTMDCVideoDataset(Dataset):
    """
    고속 KAIST 멀티카메라 Dataset
    - VideoCapture를 하나만 유지
    - Frame random access 최적화
    """
    def __init__(self, video_root, ann_root, scenario_ids,
                 transform=None, return_pid=True, use_ram=False, frame_stride=1):
        """
        video_root: D:/tar_trac/data/videos/train/
            └ s01/c01.avi ...
        ann_root:   D:/tar_trac/data/annotations/train/
        """
        self.video_root = video_root
        self.ann_root = ann_root
        self.transform = transform
        self.return_pid = return_pid
        self.use_ram = use_ram

        self.video_pool = VideoReaderPool()
        self.frame_cache = CachedFramePool(max_frames=7000)

        # 인덱스 생성
        self.index = self._build_index(scenario_ids, frame_stride=frame_stride)

        # 원하는 경우: 모든 영상을 RAM에 올리기 (가능하면 매우 빠름)
        if self.use_ram:
            self._preload_videos_into_ram()

    # -------------------------------------------------------
    # Index builder
    # -------------------------------------------------------
    def _build_index(self, scenario_ids, frame_stride=1):
        index = []
        for s in scenario_ids:
            video_dir = os.path.join(self.video_root, s)
            ann_dir   = os.path.join(self.ann_root, s)
            if not os.path.isdir(video_dir):
                continue

            for cam_file in sorted(os.listdir(video_dir)):
                if not cam_file.endswith(".avi"):
                    continue

                cam_id = cam_file.replace(".avi", "")
                video_path = os.path.join(video_dir, cam_file)
                cam_ann_dir = os.path.join(ann_dir, cam_id)
                if not os.path.isdir(cam_ann_dir):
                    continue

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # ---------- ② list annotation files ----------
                json_files = sorted([
                    f for f in os.listdir(cam_ann_dir)
                    if f.endswith(".json")
                ])

                # ---------- ③ build safe index ----------
                for jf in json_files:
                    try:
                        # Extract frame_id from filename
                        frame_id = int(jf.split("_")[-1].replace(".json", ""))
                    except:
                        # annotation 이름 이상하면 skip
                        continue

                    if frame_id < 0:
                        continue

                    # skip annotations that exceed actual video length
                    if frame_id >= total_frames:
                        # debug 출력(optional)
                        # print(f"[WARN] Skip annotation (out of range): {s}/{cam_id}/{jf} (frame={frame_id} >= total={total_frames})")
                        continue

                    if frame_id % frame_stride != 0:
                        continue

                    ann_path = os.path.join(cam_ann_dir, jf)

                    index.append({
                        "scenario": s,
                        "cam_id": cam_id,
                        "frame_id": frame_id,
                        "video_path": video_path,
                        "ann_path": ann_path
                    })
        return index

    def __len__(self):
        return len(self.index)

    # -------------------------------------------------------
    # Optional: preload into RAM
    # -------------------------------------------------------
    def _preload_videos_into_ram(self):
        print(">>> Preloading videos into RAM...")
        video_groups = defaultdict(list)
        for item in self.index:
            video_groups[item["video_path"]].append(item["frame_id"])

        self.ram_video = {}

        for video_path, frame_ids in video_groups.items():
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frames = {}
            for fid in frame_ids:
                if fid < 0 or fid >= total:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, f = cap.read()
                if not ret:
                    continue
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                frames[fid] = f

            cap.release()
            self.ram_video[video_path] = frames

        print(">>> RAM preload complete.")

    # -------------------------------------------------------
    # Frame loader
    # -------------------------------------------------------
    def _load_frame(self, video_path, frame_id):
        key = (video_path, frame_id)

        # RAM 모드
        if self.use_ram:
            return self.ram_video[video_path][frame_id]

        # 캐시 모드
        cached = self.frame_cache.get(key)
        if cached is not None:
            return cached

        # VideoCapture 재사용
        cap = self.video_pool.get(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"frame load failed: {video_path}, {frame_id}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_cache.put(key, frame)
        return frame

    # -------------------------------------------------------
    # Annotation loader
    # -------------------------------------------------------
    def _load_ann(self, ann_path):
        with open(ann_path, "r") as f:
            ann = json.load(f)

        boxes, labels, tids, pids = [], [], [], []
        for obj in ann.get("objects", []):
            if obj.get("label") != "person":
                continue

            pos = obj["position"][0]
            x1 = pos["x"]
            y1 = pos["y"]
            x2 = pos["x"] + pos["width"]
            y2 = pos["y"] + pos["height"]

            boxes.append([x1, y1, x2, y2])
            labels.append(1)
            tids.append(int(obj["track_id"]))

            attrs = obj.get("attributes", [])
            if len(attrs) > 0 and "pid" in attrs[0]:
                pids.append(int(attrs[0]["pid"]))
            else:
                pids.append(-1)

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(tids, dtype=torch.long),
            torch.tensor(pids, dtype=torch.long),
        )

    # -------------------------------------------------------
    # __getitem__
    # -------------------------------------------------------
    def __getitem__(self, idx):
        item = self.index[idx]

        frame = self._load_frame(item["video_path"], item["frame_id"])
        boxes, labels, tids, pids = self._load_ann(item["ann_path"])

        if self.transform:
            frame = self.transform(frame)

        return frame, boxes, labels, tids, pids, {
            "scenario": item["scenario"],
            "cam_id": item["cam_id"],
            "frame_id": item["frame_id"]
        }
