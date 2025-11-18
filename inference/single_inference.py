import torch
from tracking.single_camera import SingleCameraTracker

class SingleCameraInference:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def run(self, loader):
        self.model.eval()
        trackers = {}
        all_results = []

        for frames, boxes, labels, tids, pids, metas in loader:
            frames = frames.to(self.device)
            meta = metas[0]
            cam_id = meta["cam_id"]
            scenario = meta["scenario"]
            frame_id = int(meta["frame_id"])

            if cam_id not in trackers:
                trackers[cam_id] = SingleCameraTracker()

            tracker = trackers[cam_id]
            out = self.model(frames)[0]

            results = tracker.update(
                out["boxes"].to(self.device),
                out["scores"].to(self.device),
                out["labels"].to(self.device),
                out["embeds"].to(self.device),
                frame_id
            )

            for r in results:
                all_results.append({
                    "scenario": scenario,
                    "cam_id": cam_id,
                    "frame_id": frame_id,
                    "track_id": int(r["track_id"]),
                    "bbox": r["bbox"].cpu().tolist(),
                })

        return all_results
