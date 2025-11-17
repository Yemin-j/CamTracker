class GlobalIDMapper:

    def build_mapping(self, associations):
        mapping = {}
        for group in associations:
            gid = group["global_id"]
            for t in group["members"]:
                key = (t["scenario"], t["cam_id"], t["track_id"])
                mapping[key] = gid
        return mapping

    def apply(self, all_tracks, mapping):
        for r in all_tracks:
            key = (r["scenario"], r["cam_id"], r["track_id"])
            r["global_id"] = mapping.get(key, -1)
        return all_tracks
