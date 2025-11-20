# inference/export_utils.py

import os
import json
import cv2
import numpy as np


def save_mot_txt(results, save_path):
    """
    results: list of dict
        {
            "scenario": s,
            "cam_id": c,
            "frame_id": f,
            "global_id": gid (or track_id for single-camera),
            "bbox": [x1,y1,x2,y2]
        }

    save_path: "output/mot/results.txt"
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fw = open(save_path, 'w')

    for r in results:
        x1,y1,x2,y2 = r["bbox"]
        w = x2 - x1
        h = y2 - y1

        gid = r.get("global_id", r.get("track_id", -1))
        line = f"{r['frame_id']}, {gid}, {x1:.2f}, {y1:.2f}, {w:.2f}, {h:.2f}, -1, -1, -1, {r['cam_id']}\n"
        fw.write(line)

    fw.close()
    print(f"[EXPORT] MOT text saved to {save_path}")


def save_coco_json(results, save_path):
    """
    Convert tracking results to COCO annotation format
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    annotations = []
    images = {}
    ann_id = 1

    for r in results:
        img_key = f"{r['scenario']}_{r['cam_id']}_{r['frame_id']}"

        if img_key not in images:
            images[img_key] = {
                "file_name": img_key + ".jpg",
                "id": len(images) + 1,
                "frame_id": r["frame_id"],
                "cam_id": r["cam_id"]
            }

        x1,y1,x2,y2 = r["bbox"]
        w = x2 - x1
        h = y2 - y1

        annotations.append({
            "id": ann_id,
            "image_id": images[img_key]["id"],
            "category_id": 1,  # person
            "bbox": [x1, y1, w, h],
            "track_id": r.get("global_id", r.get("track_id", -1)),
            "iscrowd": 0
        })
        ann_id += 1

    coco_format = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "person"}
        ]
    }

    with open(save_path, "w") as f:
        json.dump(coco_format, f, indent=2)

    print(f"[EXPORT] COCO JSON saved to {save_path}")

def save_single_camera_avi(results, video_root, save_path, fps=23):
    """
    results: tracking 결과 (단일 카메라 전용)
    video_root: 원본 비디오 위치 (like "data/videos/test/s01/c01.avi")
    """

    # cam_id만 추출
    cam_id = results[0]["cam_id"]

    # 원본 영상 찾기
    video_file = None
    for root, dirs, files in os.walk(video_root):
        for f in files:
            if f.endswith(".avi") and cam_id in f:
                video_file = os.path.join(root, f)

    if video_file is None:
        raise FileNotFoundError("Cannot find original video for cam:", cam_id)

    cap = cv2.VideoCapture(video_file)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (W,H))

    frame_dict = {}
    for r in results:
        fid = r["frame_id"]
        if fid not in frame_dict:
            frame_dict[fid] = []
        frame_dict[fid].append(r)

    frame_ids = sorted(frame_dict.keys())

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        tracks = frame_dict[fid]
        for t in tracks:
            x1,y1,x2,y2 = map(int, t["bbox"])
            gid = t.get("global_id", t.get("track_id", -1))
            color = (0,255,0)
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, f"ID:{gid}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out.write(frame)

    cap.release()
    out.release()

    print(f"[EXPORT] AVI saved to {save_path}")

def merge_multi_camera_avi(results, video_root, save_path, fps=23, grid=(2,2)):
    """
    results: 전체 cameras global_tracks
    grid: (rows, cols)
    """

    # 카메라 ID 목록
    cam_ids = sorted(list(set(r["cam_id"] for r in results)))

    if len(cam_ids) > grid[0] * grid[1]:
        raise ValueError("Grid too small for number of cameras!")

    # 각 카메라별 프레임 단위로 group
    cam_track = {cid: {} for cid in cam_ids}

    for r in results:
        cid = r["cam_id"]
        fid = r["frame_id"]
        cam_track[cid].setdefault(fid, []).append(r)

    # 원본 비디오 resolution 얻기
    sample_cam = cam_ids[0]
    sample_video = None

    for root, dirs, files in os.walk(video_root):
        for f in files:
            if sample_cam in f and f.endswith(".avi"):
                sample_video = os.path.join(root, f)

    cap = cv2.VideoCapture(sample_video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_W = W * grid[1]
    out_H = H * grid[0]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (out_W, out_H))

    # frame id 전체 범위
    all_frame_ids = sorted(list(set(r["frame_id"] for r in results)))

    for fid in all_frame_ids:
        merged_frame = []

        for r_i in range(grid[0]):
            row_frames = []
            for c_i in range(grid[1]):
                idx = r_i * grid[1] + c_i
                if idx >= len(cam_ids):
                    blank = 255 * np.ones((H,W,3), dtype=np.uint8)
                    row_frames.append(blank)
                    continue

                cam = cam_ids[idx]
                video_file = None
                for root, dirs, files in os.walk(video_root):
                    for f in files:
                        if cam in f and f.endswith(".avi"):
                            video_file = os.path.join(root, f)

                cap = cv2.VideoCapture(video_file)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    frame = 255 * np.ones((H,W,3), dtype=np.uint8)

                # draw boxes
                items = cam_track[cam].get(fid, [])
                for t in items:
                    x1,y1,x2,y2 = map(int, t["bbox"])
                    gid = t.get("global_id", t.get("track_id", -1))
                    cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f"ID:{gid}", (x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                row_frames.append(frame)

            merged_frame.append(cv2.hconcat(row_frames))

        merged_frame = cv2.vconcat(merged_frame)
        out.write(merged_frame)

    out.release()
    print(f"[EXPORT] Multi-camera merged AVI saved: {save_path}")
