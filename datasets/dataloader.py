from torch.utils.data import DataLoader

from .aihub_dataloader import QDTrackVideoDataset
from .mot_collate_fn import QDTrackCollateFn

def data_loader(video_root, ann_root, scenario, transform, batch_size, frame_stride=1, ):
    dataset = QDTrackVideoDataset(
        video_root=video_root,
        ann_root=ann_root,
        scenario_ids=scenario,
        transform=transform,
        use_ram=False,
        frame_stride=frame_stride,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=custom_collate_fn()
    )
    return loader