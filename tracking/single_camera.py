# tracking/single_camera.py
import torch
import torch.nn.functional as F

def iou(boxes1, boxes2):
    """
    boxes1: (M,4), boxes2: (N,4)
    return: (M,N)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)))
    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    xa1 = torch.max(x11[:, None], x21[None, :])
    ya1 = torch.max(y11[:, None], y21[None, :])
    xa2 = torch.min(x12[:, None], x22[None, :])
    ya2 = torch.min(y12[:, None], y22[None, :])

    inter_w = (xa2 - xa1).clamp(min=0)
    inter_h = (ya2 - ya1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)

class TrackState:
    def __init__(self, track_id, bbox, embed, frame_idx):
        self.id = track_id
        self.bbox = bbox
        self.embed = embed
        self.last_frame = frame_idx

class SingleCameraTracker:
    def __init__(self,
                 max_age=30,
                 sim_thresh=0.5,
                 iou_thresh=0.1,
                 alpha=0.8):
        """
        max_age: 트랙 유지 프레임 수
        sim_thresh: matching threshold (similarity)
        iou_thresh: IoU gating threshold
        alpha: sim = alpha*cosine + (1-alpha)*IoU
        """
        self.max_age = max_age
        self.sim_thresh = sim_thresh
        self.iou_thresh = iou_thresh
        self.alpha = alpha

        self.tracks = {}  # track_id -> TrackState
        self.next_id = 1

    def _compute_similarity(self, track_embeds, track_boxes, det_embeds, det_boxes):
        # cosine
        if track_embeds.numel() == 0 or det_embeds.numel() == 0:
            return torch.zeros((track_embeds.size(0), det_embeds.size(0)))
        t_emb = F.normalize(track_embeds, dim=1)
        d_emb = F.normalize(det_embeds, dim=1)
        cos_sim = t_emb @ d_emb.t()   # (T,N)

        # IoU
        iou_mat = iou(track_boxes, det_boxes)  # (T,N)

        sim = self.alpha * cos_sim + (1 - self.alpha) * iou_mat
        return sim

    def update(self, det_boxes, det_scores, det_labels, det_embeds, frame_idx):
        """
        det_boxes: (N,4)
        det_scores: (N,)
        det_labels: (N,)
        det_embeds: (N,D)
        """
        device = det_boxes.device
        # 1) 살아있는 tracks만 후보
        active_ids = [tid for tid, st in self.tracks.items()
                      if frame_idx - st.last_frame <= self.max_age]

        track_boxes = torch.stack([self.tracks[tid].bbox for tid in active_ids], dim=0) if active_ids else torch.empty((0,4), device=device)
        track_embeds = torch.stack([self.tracks[tid].embed for tid in active_ids], dim=0) if active_ids else torch.empty((0,det_embeds.size(1)), device=device)

        # 2) 유사도 계산
        sim = self._compute_similarity(track_embeds, track_boxes, det_embeds, det_boxes)  # (T,N)
        matches = []
        unmatched_tracks = set(active_ids)
        unmatched_dets = set(range(det_boxes.size(0)))

        # 3) greedy matching
        if sim.numel() > 0:
            T, N = sim.shape
            sim_flat = sim.view(-1)
            sorted_idx = torch.argsort(sim_flat, descending=True)
            for idx in sorted_idx:
                t_idx = idx // N
                d_idx = idx % N
                tid = active_ids[t_idx]
                if tid not in unmatched_tracks:
                    continue
                if d_idx not in unmatched_dets:
                    continue
                if sim[t_idx, d_idx] < self.sim_thresh:
                    break
                # IoU gating (optional)
                if iou(track_boxes[t_idx:t_idx+1], det_boxes[d_idx:d_idx+1]).item() < self.iou_thresh:
                    continue
                matches.append((tid, int(d_idx)))
                unmatched_tracks.remove(tid)
                unmatched_dets.remove(d_idx)

        # 4) matched tracks 업데이트
        for tid, d_idx in matches:
            self.tracks[tid].bbox = det_boxes[d_idx]
            self.tracks[tid].embed = det_embeds[d_idx]
            self.tracks[tid].last_frame = frame_idx

        # 5) unmatched detection → 새 track 생성
        for d_idx in unmatched_dets:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = TrackState(
                tid,
                det_boxes[d_idx],
                det_embeds[d_idx],
                frame_idx
            )

        # 6) 이번 프레임 결과 반환
        results = []
        for tid, st in self.tracks.items():
            if st.last_frame == frame_idx:
                results.append(dict(
                    track_id=tid,
                    bbox=st.bbox,
                    frame_idx=frame_idx,
                    embed=st.embed
                ))
        return results
