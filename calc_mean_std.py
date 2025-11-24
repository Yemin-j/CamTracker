import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.aihub_dataloader import AIMTMDCVideoDataset
from datasets.mot_collate_fn import custom_collate_fn
import torchvision.transforms as T

def calc_mean_std(video_root, ann_root, scenarios, sample_ratio=5):
    transform = T.ToTensor()

    dataset = AIMTMDCVideoDataset(
        video_root=video_root,
        ann_root=ann_root,
        scenario_ids=scenarios,
        transform=transform,
        use_ram=False,
        frame_stride=sample_ratio  # sampling ratio
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn())

    mean = torch.zeros(3)
    std  = torch.zeros(3)
    total = 0

    for frames, _, _, _, _, _ in tqdm(loader):
        img = frames[0]   # (3,H,W)
        total += 1

        mean += img.mean(dim=[1,2])
        std  += img.std(dim=[1,2])

    mean /= total
    std  /= total

    return mean, std


video_root = "D:/tar_trac/data/videos/train"
ann_root   = "D:/tar_trac/data/annotations/train"

mean, std = calc_mean_std(video_root, ann_root, scenarios=["s01"], sample_ratio=50)
print("Mean =", mean)
print("Std  =", std)
