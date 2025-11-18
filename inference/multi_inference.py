import torch
from tracking.single_camera import SingleCameraTracker
from tracking.tracklet_builder import TrackletBuilder
from tracking.feature_bank import TrackletFeatureExtractor
from tracking.mcta import MultiCameraAssociator
from tracking.global_id_mapper import GlobalIDMapper

class MultiCameraInference:

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def run(self, loader):
        self.model.eval()
        trackers = {}
        all_tracks = []

        # 1) Single-camera tracking
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
                all_tracks.append({
                    "scenario": scenario,
                    "cam_id": cam_id,
                    "frame_id": frame_id,
                    "track_id": int(r["track_id"]),
                    "bbox": r["bbox"].cpu().tolist(),
                    "embed": r["embed"].cpu().tolist(),
                })

        # 2) Tracklet 만들기
        builder = TrackletBuilder()
        tracklets = builder.build(all_tracks)

        # 3) tracklet embedding
        extractor = TrackletFeatureExtractor()
        extractor.extract(tracklets)

        # 4) MCTA
        associator = MultiCameraAssociator(mode="greedy", thresh=0.5)
        associations = associator.associate(tracklets)

        # 5) global ID mapping
        mapper = GlobalIDMapper()
        mapping = mapper.build_mapping(associations)
        global_tracks = mapper.apply(all_tracks, mapping)

        return global_tracks
