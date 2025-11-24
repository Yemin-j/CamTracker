import torch


class QDTrackCollateFn:
    """
    Collate function for QDTrack temporal pair dataset

    Handles batching of (key_frame, ref_frame) pairs with variable-length boxes
    """

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples from dataset __getitem__
                   Each tuple: (frame_key, frame_ref, boxes_key, boxes_ref,
                               ids_key, ids_ref, meta)

        Returns:
            frames_key: (B,3,H,W) stacked key frames
            frames_ref: (B,3,H,W) stacked reference frames
            boxes_key: List[Tensor(N_i,4)] list of boxes for each key frame
            boxes_ref: List[Tensor(M_i,4)] list of boxes for each ref frame
            ids_key: List[Tensor(N_i,)] list of track IDs for key frames
            ids_ref: List[Tensor(M_i,)] list of track IDs for ref frames
            metas: List[dict] metadata for each sample
        """
        frames_key = []
        frames_ref = []
        boxes_key = []
        boxes_ref = []
        ids_key = []
        ids_ref = []
        metas = []

        for item in batch:
            (frame_k, frame_r, box_k, box_r, id_k, id_r, meta) = item

            frames_key.append(frame_k)  # (3,H,W)
            frames_ref.append(frame_r)  # (3,H,W)
            boxes_key.append(box_k)  # (N,4)
            boxes_ref.append(box_r)  # (M,4)
            ids_key.append(id_k)  # (N,)
            ids_ref.append(id_r)  # (M,)
            metas.append(meta)

        # Stack frames (fixed size)
        frames_key = torch.stack(frames_key, dim=0)  # (B,3,H,W)
        frames_ref = torch.stack(frames_ref, dim=0)  # (B,3,H,W)

        # Keep boxes and IDs as lists (variable length per image)
        return frames_key, frames_ref, boxes_key, boxes_ref, ids_key, ids_ref, metas