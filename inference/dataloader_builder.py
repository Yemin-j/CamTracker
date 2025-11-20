from torch.utils.data import DataLoader

from datasets.mot_dataloader import AIMTMDCVideoDataset
from datasets.mot_collate_fn import custom_collate_fn

def test_data_loader(video_root, ann_root, scenarios, transform, batch_size=1, frame_stride=1):
    dataset = AIMTMDCVideoDataset(
        video_root=video_root,
        ann_root=ann_root,
        scenario_ids=scenarios,
        transform=transform,
        use_ram=False,
        frame_stride=frame_stride,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=custom_collate_fn()
    )
    return loader
