# test.py

import torch
from torchvision import transforms as T

from models.detector import DetectorWithReID
from datasets.dataloader import data_loader
from inference.dataloader_builder import test_data_loader
from inference.single_inference import SingleCameraInference
from inference.multi_inference import MultiCameraInference
from inference.export_utils import save_mot_txt, save_coco_json, save_single_camera_avi, merge_multi_camera_avi


def main(mode="single"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 준비
    model = DetectorWithReID.build_default(num_ids=500).to(device)
    model.load_state_dict(torch.load("checkpoint.pth", map_location=device))

    # Inference dataloader
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    infer_loader = test_data_loader(
        video_root="D:/tar_trac/data/videos/test",
        ann_root="D:/tar_trac/data/annotations/test",
        scenarios=["s01", "s02"],
        transform=transform,
        batch_size=1
    )

    if mode == "single":
        engine = SingleCameraInference(model, device)
        results = engine.run(infer_loader)
        print("SINGLE CAMERA RESULTS:", results[:5])

        save_mot_txt(results, "results/test/mot.txt")
        save_coco_json(results, "results/test/coco.json")
        save_single_camera_avi(results, video_root, "results/test/c01.avi")

    elif mode == "multi":
        engine = MultiCameraInference(model, device)
        global_tracks = engine.run(infer_loader)
        print("MULTI CAMERA TRACKING:", global_tracks[:5])

        save_mot_txt(global_tracks, "results/test/mot_global.txt")
        save_coco_json(global_tracks, "results/test/coco_global.json")
        merge_multi_camera_avi(global_tracks, video_root, "results/test/multi.avi")


if __name__ == "__main__":
    main(mode="multi")
    # main(mode="single")
