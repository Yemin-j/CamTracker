# tracking/single_camera.py
import torch
import torch.nn.functional as F


def iou(boxes1, boxes2):
    """
    boxes1: (M,4), boxes2: (N,4)
    return: (M,N)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), device=boxes1.device)

    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # pairwise max/min
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
    def __init__(self, last_frame, box, embed, score):
        self.last_frame = last_frame
        self.box = box      # CPU tensor (4,)
        self.embed = embed  # CPU tensor (D,)
        self.score = score  # float


class SingleCameraTracker:
    def __init__(
        self,
        max_age=5,
        sim_thresh=0.0,
        iou_thresh=0.05,
        alpha=0.3,
        score_thresh=0.3,
    ):
        """
        max_age: 트랙 유지 프레임 수
        sim_thresh: similarity gate (0으로 두면 사실상 사용 안 함)
        iou_thresh: IoU gating threshold
        alpha: sim = alpha*cosine + (1-alpha)*IoU
        score_thresh: det score threshold (낮은 det 필터링)
        """
        self.max_age = max_age
        self.sim_thresh = sim_thresh
        self.iou_thresh = iou_thresh
        self.alpha = alpha
        self.score_thresh = score_thresh

        self.tracks = {}  # track_id -> TrackState
        self.next_id = 1

    def update(self, det_boxes, det_scores, det_labels, det_embeds, frame_idx):
        """
        det_boxes: (N,4)
        det_scores: (N,)
        det_labels: (N,)
        det_embeds: (N,D)
        """
        device = det_boxes.device

        # 0) score filtering
        keep = det_scores >= self.score_thresh
        det_boxes = det_boxes[keep]
        det_scores = det_scores[keep]
        det_labels = det_labels[keep]
        det_embeds = det_embeds[keep]

        N = det_boxes.size(0)
        if N == 0:
            # 아무 detection도 없으면 track만 정리하고 끝
            # 오래된 트랙 삭제
            to_delete = [
                tid for tid, st in self.tracks.items()
                if frame_idx - st.last_frame > self.max_age
            ]
            for tid in to_delete:
                del self.tracks[tid]
            return []

        # 1) 오래된 트랙 삭제 (tracks dict 크기 제한)
        to_delete = [
            tid for tid, st in self.tracks.items()
            if frame_idx - st.last_frame > self.max_age
        ]
        for tid in to_delete:
            del self.tracks[tid]

        active_ids = list(self.tracks.keys())
        T = len(active_ids)

        results = []

        # 2) 이전 트랙이 하나도 없으면 전부 새 트랙
        if T == 0:
            for i in range(N):
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = TrackState(
                    last_frame=frame_idx,
                    box=det_boxes[i].detach().cpu(),
                    embed=det_embeds[i].detach().cpu(),
                    score=float(det_scores[i]),
                )
                results.append({
                    "track_id": tid,
                    "bbox": det_boxes[i].detach().cpu(),
                    "score": float(det_scores[i]),
                })
            return results

        # 3) track state 텐서로 변환
        track_boxes = torch.stack(
            [self.tracks[tid].box.to(device) for tid in active_ids], dim=0
        )
        track_embs = torch.stack(
            [self.tracks[tid].embed.to(device) for tid in active_ids], dim=0
        )

        T = track_boxes.size(0)

        # 4) similarity 계산
        with torch.no_grad():
            t_emb = F.normalize(track_embs, dim=1)
            d_emb = F.normalize(det_embeds, dim=1)

            cos_sim = t_emb @ d_emb.t()  # (T,N)
            iou_mat = iou(track_boxes, det_boxes)  # (T,N)
            sim = self.alpha * cos_sim + (1.0 - self.alpha) * iou_mat  # (T,N)

        # 5) greedy matching
        flat = sim.flatten()  # (T*N,)
        sorted_idx = torch.argsort(flat, descending=True)

        assigned_tracks = set()
        assigned_dets = set()
        matches = []

        for idx_flat in sorted_idx:
            tid_idx = int(idx_flat // N)  # track index
            det_idx = int(idx_flat % N)   # det index

            if tid_idx in assigned_tracks or det_idx in assigned_dets:
                continue

            # IoU gating
            if iou_mat[tid_idx, det_idx].item() < self.iou_thresh:
                continue

            # sim gating (필요 없으면 sim_thresh=0)
            if sim[tid_idx, det_idx].item() < self.sim_thresh:
                continue

            assigned_tracks.add(tid_idx)
            assigned_dets.add(det_idx)
            matches.append((tid_idx, det_idx))

            if len(assigned_tracks) == T or len(assigned_dets) == N:
                break

        used_det = set()

        # 6) 매칭된 트랙 업데이트
        for tid_idx, det_idx in matches:
            track_id = active_ids[tid_idx]
            self.tracks[track_id].last_frame = frame_idx
            self.tracks[track_id].box = det_boxes[det_idx].detach().cpu()
            self.tracks[track_id].embed = det_embeds[det_idx].detach().cpu()
            self.tracks[track_id].score = float(det_scores[det_idx])

            used_det.add(det_idx)

            results.append({
                "track_id": track_id,
                "bbox": det_boxes[det_idx].detach().cpu(),
                "score": float(det_scores[det_idx]),
            })

        # 7) 매칭되지 않은 detection은 새 트랙 생성
        for det_idx in range(N):
            if det_idx in used_det:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = TrackState(
                last_frame=frame_idx,
                box=det_boxes[det_idx].detach().cpu(),
                embed=det_embeds[det_idx].detach().cpu(),
                score=float(det_scores[det_idx]),
            )
            results.append({
                "track_id": tid,
                "bbox": det_boxes[det_idx].detach().cpu(),
                "score": float(det_scores[det_idx]),
            })

        return results
