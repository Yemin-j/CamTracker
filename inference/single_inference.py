# inference/single_inference.py
import torch
from tqdm import tqdm
from tracking.single_camera import SingleCameraTracker


class SingleCameraInference:
    def __init__(self, model, device, logger=None):
        self.model = model
        self.device = device
        self.logger = logger

    @torch.no_grad()
    def run(self, loader):
        """
        loader batch format:
          - temporal(QDTrackCollateFn): (frames_key, frames_ref, boxes_key, boxes_ref, ids_key, ids_ref, metas)
          - single frame(simple_collate): (frames, boxes, labels, tids, pids, metas)
        """
        if self.logger:
            self.logger.info("Starting Single Camera Tracking...")

        self.model.eval()
        trackers = {}
        all_results = []

        iterator = enumerate(loader)
        if self.logger:
            iterator = tqdm(iterator, total=len(loader), desc="SingleCam Tracking")

        for batch_idx, batch in iterator:
            # --- batch unpack ---
            if len(batch) == 7:
                frames_key, frames_ref, boxes_key, boxes_ref, ids_key, ids_ref, metas = batch
                frames = frames_key  # tracking은 key-frame만 사용
                meta = metas[0]
            else:
                # single-frame format
                frames, boxes, labels, tids, pids, metas = batch
                meta = metas[0]

            # move frames to GPU
            frames = frames.to(self.device, non_blocking=True)

            # meta fields
            seq_id = meta.get("seq", meta.get("scenario", "unknown"))
            cam_id = meta.get("cam_id", seq_id)

            if "frame_id" in meta:
                frame_id = int(meta["frame_id"])
            elif "key_frame" in meta:
                frame_id = int(meta["key_frame"])
            elif "key_frame_id" in meta:
                frame_id = int(meta["key_frame_id"])
            else:
                frame_id = 0

            # tracker per camera
            if cam_id not in trackers:
                trackers[cam_id] = SingleCameraTracker()
            tracker = trackers[cam_id]

            # model forward
            outputs = self.model(frames)[0]

            # tracking update
            results = tracker.update(
                outputs["boxes"].to(self.device),
                outputs["scores"].to(self.device),
                outputs["labels"].to(self.device),
                outputs["embeds"].to(self.device),
                frame_id,
            )

            # save results
            for r in results:
                all_results.append({
                    "scenario": seq_id,
                    "cam_id": cam_id,
                    "frame_id": frame_id,
                    "track_id": int(r["track_id"]),
                    "bbox": r["bbox"].tolist(),  # CPU tensor -> list
                })

        return all_results
