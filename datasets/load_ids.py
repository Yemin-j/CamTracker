import os

def list_visdrone_sequences(video_root):
    """
    VisDrone2019-MOT에서 sequences 폴더 아래 모든 sequence ID 자동 수집.
    예: uav0000013_00000
    """
    if not os.path.isdir(video_root):
        raise RuntimeError(f"Invalid video_root: {video_root}")

    seq_ids = []
    for name in sorted(os.listdir(video_root)):
        full = os.path.join(video_root, name)
        if os.path.isdir(full):
            seq_ids.append(name)

    return seq_ids
