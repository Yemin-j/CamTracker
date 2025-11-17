# main_train.py
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.optim import Adam

from datasets.mot_dataloader import AIMTMDCVideoDataset
from datasets.mot_collate_fn import custom_collate_fn

from models.res50_backbone import FPNBackbone
from models.rpn_head import RPNHead
from models.roi import RoIAlignLayer
from models.bbox_head import BBoxHead
from models.reid_head import ReIDHead
from models.detector import DetectorWithReID

from tracking.single_camera import SingleCameraTracker  # 학습에는 안 써도 됨
from tracking.multi_camera_pipeline import MultiCameraTrackingPipeline

from hook.tracking_eval_hook import TrackingEvalHook

def build_detector_with_reid(device, num_ids):
    backbone = FPNBackbone()
    rpn_head = RPNHead(
        in_channels=256,
        num_anchors=9,
        strides=[8,16,32,64],
        scales=[4,8,16],
        ratios=[0.5,1.0,2.0],
        pre_nms_topk=1000,
        post_nms_topk=200,
        nms_thresh=0.7
    )
    roi_align = RoIAlignLayer(output_size=7)
    bbox_head = BBoxHead(in_channels=256, fc_dim=1024, num_classes=1, score_thresh=0.0)
    reid_head = ReIDHead(in_channels=256, embed_dim=256, num_ids=num_ids)
    model = DetectorWithReID(backbone, rpn_head, roi_align, bbox_head, reid_head)
    model.to(device)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    video_root = r"D:/tar_trac/data/videos/train"
    ann_root   = r"D:/tar_trac/data/annotations/train"

    train_scenarios = [f"s{i:02d}" for i in range(1, 2)]
    val_scenarios = [f"s{i:02d}" for i in range(19, 20)]

    train_dataset = AIMTMDCVideoDataset(
        video_root=video_root,
        ann_root=ann_root,
        scenario_ids=train_scenarios,
        transform=transform,
        use_ram=False
    )
    val_dataset = AIMTMDCVideoDataset(
        video_root=video_root,
        ann_root=ann_root,
        scenario_ids=val_scenarios,
        transform=transform,
        use_ram=False
    )
    # TODO: pid → 0..num_ids-1 매핑 필요 (여기선 num_ids를 적당히 넣어둔다)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=0, collate_fn=custom_collate_fn(),)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=custom_collate_fn(),)

    num_ids = 200  # TODO: dataset 전체 pid 기준으로 실제 숫자 세팅
    model = build_detector_with_reid(device, num_ids)
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Hook 등록
    eval_hook = TrackingEvalHook(val_loader, SingleCameraTracker, device, interval=1)
    num_epochs = 2

    model.train()
    for epoch in range(num_epochs):
        model.train()
        for iter_i, batch in enumerate(train_loader):
            frame, boxes, labels, tids, pids, meta = batch

            frame = frame.to(device)

            gt_boxes = [b.to(device) for b in boxes]
            gt_labels = [l.to(device) for l in labels]
            gt_ids = [p.to(device) for p in pids]

            optimizer.zero_grad()
            losses = model.forward_train(frame, gt_boxes, gt_labels, gt_ids, epoch=epoch, warmup_epochs=3)
            loss = losses["loss_total"]
            loss.backward()
            optimizer.step()

            if iter_i % 10 == 0:
                print(f"[Train] epoch {epoch + 1}, iter {iter_i}, "
                      f"total={loss.item():.4f}, "
                      f"cls={losses['loss_cls'].item():.4f}, "
                      f"reg={losses['loss_reg'].item():.4f}, "
                      f"reid={losses['loss_reid'].item():.4f}")

        # epoch 끝나면 자동 검증
        eval_hook.after_epoch(epoch, model)

if __name__ == "__main__":
    main()
