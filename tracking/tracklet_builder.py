from collections import defaultdict

class TrackletBuilder:
    def build(self, all_tracks):
        """
        all_tracks: [
            {"scenario":..., "cam_id":..., "frame_id":..., "track_id":..., "bbox":..., "embed":...}
        ]
        return: tracklets list
        """

        grouped = defaultdict(list)

        # (scenario, cam, track_id) 단위로 모으기
        for r in all_tracks:
            key = (r["scenario"], r["cam_id"], r["track_id"])
            grouped[key].append(r)

        tracklets = []
        for (scenario, cam_id, tid), records in grouped.items():
            records = sorted(records, key=lambda x: x["frame_id"])
            frames = [r["frame_id"] for r in records]
            bboxes = [r["bbox"] for r in records]
            embeds = [r["embed"] for r in records]

            tracklets.append({
                "scenario": scenario,
                "cam_id": cam_id,
                "track_id": tid,
                "frames": frames,
                "bboxes": bboxes,
                "embeds": embeds,
                "start_frame": frames[0],
                "end_frame": frames[-1],
            })

        return tracklets