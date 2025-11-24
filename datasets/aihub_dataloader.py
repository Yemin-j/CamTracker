import os
import json
from json.decoder import JSONDecodeError
import cv2
import torch
import random
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict


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


class LRUFrameCache:
    """
    LRU cache for frames with memory limit

    MEMORY SAFETY:
    - Default max_size=200 frames (~1.2GB for 960x540 frames)
    - Automatically evicts oldest frames when limit reached
    - Clear() method for manual cleanup between epochs
    """

    def __init__(self, max_size=200):  # ✅ REDUCED from 1000 to 200
        self.cache = OrderedDict()
        self.max_size = max_size
        self._cache_hits = 0
        self._cache_misses = 0

    def get(self, key):
        if key not in self.cache:
            self._cache_misses += 1
            return None
        self._cache_hits += 1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        """Clear all cached frames - call between epochs to free memory"""
        self.cache.clear()

    def get_stats(self):
        """Get cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }


class QDTrackVideoDataset(Dataset):
    """
    QDTrack-style KAIST 멀티카메라 Dataset with Temporal Pair Sampling

    MEMORY OPTIMIZATIONS:
    - Reduced frame cache size (200 frames ~1.2GB)
    - Optional cache clearing between epochs
    - use_ram=False by default (don't preload everything)
    - Frames loaded on-demand and cached with LRU eviction
    """

    def __init__(self, video_root, ann_root, scenario_ids,
                 transform=None,
                 use_ram=False,
                 frame_stride=1,
                 temporal_offset_range=3,
                 min_track_length=4,
                 cache_size=200):  # Configurable cache size
        """
        Args:
            cache_size: Number of frames to cache (default 200 ~1.2GB)
                       Set to 0 to disable caching
        """
        self.video_root = video_root
        self.ann_root = ann_root
        self.transform = transform
        self.use_ram = use_ram
        self.temporal_offset_range = temporal_offset_range

        self.video_pool = VideoReaderPool()
        self.frame_cache = LRUFrameCache(max_size=cache_size) if cache_size > 0 else None

        # Build video-level index with track information
        self.video_index, self.track_info, self.pairs = self._build_video_index_and_pairs(
            scenario_ids, frame_stride, min_track_length)

        if self.use_ram:
            print("[WARNING] use_ram=True will load all videos into RAM!")
            print("This may consume 10+ GB of memory.")
            self._preload_videos_into_ram()

    def _build_video_index_and_pairs(self, scenario_ids, frame_stride, min_track_length):
        """
        QDTrack 스타일:
        - video_index: 비디오별 frame / track 정보
        - pairs: 공통 track_id가 있는 (video_key, key_frame_id, ref_frame_id) 리스트
        """
        video_sequences = defaultdict(lambda: defaultdict(list))

        # 1) 프레임별 기본 정보 수집 (기존과 동일)
        for s in scenario_ids:
            video_dir = os.path.join(self.video_root, s)
            ann_dir = os.path.join(self.ann_root, s)
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

                json_files = sorted([
                    f for f in os.listdir(cam_ann_dir)
                    if f.endswith(".json")
                ])

                video_key = (s, cam_id, video_path)

                for jf in json_files:
                    try:
                        frame_id = int(jf.split("_")[-1].replace(".json", ""))
                    except:
                        continue

                    if frame_id < 0 or frame_id >= total_frames:
                        continue
                    if frame_id % frame_stride != 0:
                        continue

                    ann_path = os.path.join(cam_ann_dir, jf)

                    video_sequences[video_key][frame_id] = {
                        "video_path": video_path,
                        "ann_path": ann_path,
                        "scenario": s,
                        "cam_id": cam_id,
                    }

        # 2) video_index, track_ids_per_frame, pairs 생성
        video_index = []
        track_info = {}
        pairs = []

        for video_key, frames_dict in video_sequences.items():
            sorted_frame_ids = sorted(frames_dict.keys())
            if len(sorted_frame_ids) < min_track_length:
                continue

            # frame별 track id 집합
            track_ids_per_frame = {}
            for frame_id in sorted_frame_ids:
                ann_path = frames_dict[frame_id]["ann_path"]
                track_ids = set(self._get_track_ids_from_ann(ann_path))
                track_ids_per_frame[frame_id] = track_ids

            video_index.append({
                "video_key": video_key,
                "frames": frames_dict,
                "sorted_frame_ids": sorted_frame_ids,
                "track_ids_per_frame": track_ids_per_frame
            })
            track_info[video_key] = track_ids_per_frame

            # 3) 이 비디오에서 공통 ID가 있는 frame pair를 모두 수집
            num_frames = len(sorted_frame_ids)
            max_offset = self.temporal_offset_range

            for i in range(num_frames):
                key_frame_id = sorted_frame_ids[i]
                key_ids = track_ids_per_frame[key_frame_id]
                if not key_ids:
                    continue  # 사람 없는 프레임은 스킵

                # 주변 frame을 살펴보며 공통 ID가 있으면 pair 추가
                for j in range(max(0, i - max_offset), min(num_frames, i + max_offset + 1)):
                    if i == j:
                        continue
                    ref_frame_id = sorted_frame_ids[j]
                    ref_ids = track_ids_per_frame[ref_frame_id]
                    if not ref_ids:
                        continue

                    common_ids = key_ids.intersection(ref_ids)
                    if len(common_ids) > 0:
                        pairs.append({
                            "video_key": video_key,
                            "key_frame_id": key_frame_id,
                            "ref_frame_id": ref_frame_id
                        })

        return video_index, track_info, pairs

    def _get_track_ids_from_ann(self, ann_path):
        """Extract track IDs from annotation file"""
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
        except:
            return []

        track_ids = []
        for obj in ann.get("objects", []):
            if obj.get("label") == "person":
                track_ids.append(int(obj["track_id"]))
        return track_ids

    def __len__(self):
        return len(self.pairs)

    def _preload_videos_into_ram(self):
        """WARNING: This loads ALL videos into RAM - use carefully!"""
        print(">>> Preloading videos into RAM...")
        self.ram_video = {}

        processed_videos = set()
        for video_seq in self.video_index:
            frames_dict = video_seq["frames"]

            for frame_id, frame_info in frames_dict.items():
                video_path = frame_info["video_path"]

                if video_path in processed_videos:
                    continue

                if video_path not in self.ram_video:
                    self.ram_video[video_path] = {}

                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.ram_video[video_path][frame_id] = frame

                processed_videos.add(video_path)

        print(">>> RAM preload complete.")

    def _load_frame(self, video_path, frame_id):
        """
        Load frame with caching
        Returns:
            frame_tensor: (3, resize_h, resize_w)
            ※ orig_w, orig_h 리턴 제거 (요청사항)
        """

        key = (video_path, frame_id)

        # RAM mode
        if self.use_ram:
            if video_path in self.ram_video and frame_id in self.ram_video[video_path]:
                frame = self.ram_video[video_path][frame_id]
                resize_w, resize_h = 960, 540
                frame = cv2.resize(frame, (resize_w, resize_h))
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                return frame_tensor

        # Cache mode
        if self.frame_cache is not None:
            cached = self.frame_cache.get(key)
            if cached is not None:
                return cached

        # Load from disk
        cap = self.video_pool.get(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Frame load failed: {video_path}, frame {frame_id}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Resize here ---
        resize_w, resize_h = 960, 540
        frame = cv2.resize(frame, (resize_w, resize_h))

        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        if self.frame_cache is not None:
            self.frame_cache.put(key, frame_tensor)

        return frame_tensor

    def _load_ann(self, ann_path):
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
        except:
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)

        boxes = []
        track_ids = []

        # HARD-CODED ORIGINAL SIZE
        orig_w = 1920
        orig_h = 1080

        resize_w, resize_h = 960, 540
        sx = resize_w / orig_w
        sy = resize_h / orig_h

        for obj in ann.get("objects", []):
            if obj["label"] != "person":
                continue

            pos = obj["position"][0]
            x1 = pos["x"] * sx
            y1 = pos["y"] * sy
            x2 = (pos["x"] + pos["width"]) * sx
            y2 = (pos["y"] + pos["height"]) * sy

            boxes.append([x1, y1, x2, y2])
            track_ids.append(int(obj["track_id"]))

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(track_ids, dtype=torch.long)
        )

    def _find_valid_reference_frame(self, video_seq, key_frame_id, key_track_ids):
        """Find a valid reference frame within temporal offset range"""
        sorted_frame_ids = video_seq["sorted_frame_ids"]
        track_ids_per_frame = video_seq["track_ids_per_frame"]

        try:
            key_idx = sorted_frame_ids.index(key_frame_id)
        except ValueError:
            return None

        offset = random.randint(-self.temporal_offset_range, self.temporal_offset_range)
        if offset == 0:
            offset = 1 if random.random() > 0.5 else -1

        ref_idx = key_idx + offset
        ref_idx = max(0, min(len(sorted_frame_ids) - 1, ref_idx))

        ref_frame_id = sorted_frame_ids[ref_idx]
        ref_track_ids = track_ids_per_frame[ref_frame_id]

        common_ids = key_track_ids.intersection(ref_track_ids)

        if len(common_ids) > 0:
            return ref_frame_id

        # Fallback: search nearby
        for search_offset in range(1, self.temporal_offset_range + 1):
            for direction in [1, -1]:
                search_idx = key_idx + direction * search_offset
                if 0 <= search_idx < len(sorted_frame_ids):
                    search_frame_id = sorted_frame_ids[search_idx]
                    search_track_ids = track_ids_per_frame[search_frame_id]
                    common_ids = key_track_ids.intersection(search_track_ids)
                    if len(common_ids) > 0:
                        return search_frame_id

        return sorted_frame_ids[ref_idx]

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        video_key = pair["video_key"]
        key_frame_id = pair["key_frame_id"]
        ref_frame_id = pair["ref_frame_id"]
        # Find sequence
        video_seq = next(seq for seq in self.video_index if seq["video_key"] == video_key)
        frames_dict = video_seq["frames"]

        key_info = frames_dict[key_frame_id]
        ref_info = frames_dict[ref_frame_id]

        frame_key = self._load_frame(key_info["video_path"], key_frame_id)
        frame_ref = self._load_frame(ref_info["video_path"], ref_frame_id)

        boxes_key, ids_key = self._load_ann(key_info["ann_path"])
        boxes_ref, ids_ref = self._load_ann(ref_info["ann_path"])

        if self.transform:
            frame_key = self.transform(frame_key)
            frame_ref = self.transform(frame_ref)

        meta = {
            "scenario": key_info["scenario"],
            "cam_id": key_info["cam_id"],
            "key_frame_id": key_frame_id,
            "ref_frame_id": ref_frame_id,
            "temporal_offset": ref_frame_id - key_frame_id,
            "resize_w": 960,
            "resize_h": 540,
        }

        return frame_key, frame_ref, boxes_key, boxes_ref, ids_key, ids_ref, meta

    def clear_cache(self):
        """Clear frame cache - call between epochs to free memory"""
        if self.frame_cache is not None:
            self.frame_cache.clear()

    def get_cache_stats(self):
        """Get cache statistics"""
        if self.frame_cache is not None:
            return self.frame_cache.get_stats()
        return None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        if hasattr(self, 'video_pool'):
            self.video_pool.release_all()
        return False

    def __del__(self):
        """Destructor - cleanup video resources"""
        try:
            if hasattr(self, 'video_pool'):
                self.video_pool.release_all()
        except:
            pass