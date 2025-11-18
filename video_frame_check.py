import cv2
import matplotlib.pyplot as plt

def visualize_annotation_time(video_path, ann_frame_ids, ann_fps=23):
    """
    video_path: 원본 비디오 경로 (30fps 기준)
    ann_frame_ids: annotation frame_id 리스트 (0~7361)
    ann_fps: annotation fps (기본 23)

    결과:
      - annotation frame_id → annotation time(sec)
      - annotation frame_id → raw video time(sec)
    """

    cap = cv2.VideoCapture(video_path)
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[VIDEO] raw_fps={raw_fps}, raw_frames={raw_frames}")

    # annotation time 기준 (annotation 자체 23fps 시간)
    ann_times = [fid / ann_fps for fid in ann_frame_ids]

    # raw 영상 time 기준 (비디오에서 해당 fid번째 프레임의 시간)
    # 단, annotation frame_id 가 raw video frame index라고 "가정"하지 않고,
    # annotation이 23fps로 균등 샘플링되었다는 사실을 사용해
    # 실제 raw video time 위치는 다음과 같이 계산:
    raw_times = [fid / ann_fps for fid in ann_frame_ids]  # ☆ 핵심 포인트

    return raw_fps, raw_frames, ann_times, raw_times

def plot_annotation_time(ann_frame_ids, ann_times, raw_times):
    plt.figure(figsize=(10,5))
    plt.plot(ann_frame_ids, ann_times, label="Annotation Time (sec)", linewidth=2)
    plt.plot(ann_frame_ids, raw_times, label="Raw Video Time (sec)", linestyle="--")
    plt.xlabel("annotation frame_id")
    plt.ylabel("time (sec)")
    plt.title("Annotation Frame_id → Video Time Mapping")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_coverage(raw_frames, raw_fps, ann_frame_ids, ann_fps=23):
    raw_total_sec = raw_frames / raw_fps
    ann_total_sec = max(ann_frame_ids) / ann_fps

    plt.figure(figsize=(10,2))
    plt.hlines(1, 0, raw_total_sec, colors='gray', linewidth=10, label="Raw Video Range")
    plt.hlines(1, 0, ann_total_sec, colors='blue', linewidth=10, label="Annotation Range")
    plt.xlabel("Time (sec)")
    plt.yticks([])
    plt.title("Coverage of Annotation vs Raw Video")
    plt.legend()
    plt.show()

    print(f"Raw video length  : {raw_total_sec:.2f} sec")
    print(f"Annotation length : {ann_total_sec:.2f} sec")

if __name__=='__main__':
    video_path = "D:/tar_trac/data/videos/train/s01/카메라02.avi"

    # annotation 프레임 아이디 예시:
    ann_frame_ids = list(range(0, 7362))

    raw_fps, raw_frames, ann_times, raw_times = visualize_annotation_time(
        video_path,
        ann_frame_ids,
        ann_fps=23
    )

    plot_annotation_time(ann_frame_ids, ann_times, raw_times)
    plot_coverage(raw_frames, raw_fps, ann_frame_ids)
