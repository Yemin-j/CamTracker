import torch
import torch.nn.functional as F
import math

class MultiCameraAssociator:

    def __init__(self, mode="greedy", weights=None, thresh=0.5):
        self.mode = mode
        self.weights = weights
        self.thresh = thresh

    # ===============================
    # Distance components
    # ===============================
    def appearance_similarity(self, a, b):
        return float(F.cosine_similarity(a["feat"], b["feat"], dim=0))

    def d_app(self, a, b):
        return 1.0 - self.appearance_similarity(a, b)

    def temporal_compatible(self, a, b, max_gap=200):
        # same cam → overlap 금지
        if a["cam_id"] == b["cam_id"]:
            overlap = min(a["end_frame"], b["end_frame"]) - max(a["start_frame"], b["start_frame"])
            if overlap > 0:
                return False

        # 너무 멀리 떨어진 tracklet은 연결 금지
        gap = min(abs(a["start_frame"] - b["end_frame"]),
                  abs(b["start_frame"] - a["end_frame"]))
        return gap <= max_gap

    def d_time(self, a, b):
        c1 = (a["start_frame"] + a["end_frame"]) / 2
        c2 = (b["start_frame"] + b["end_frame"]) / 2
        return abs(c1 - c2) / 1000.0

    def d_cam(self, a, b):
        return 1.0 if a["cam_id"] == b["cam_id"] else 0.0

    def d_pos(self, a, b):
        ax1, ay1, ax2, ay2 = a["bboxes"][0]
        bx1, by1, bx2, by2 = b["bboxes"][0]
        acx = (ax1 + ax2)/2; acy = (ay1 + ay2)/2
        bcx = (bx1 + bx2)/2; bcy = (by1 + by2)/2
        dist = math.sqrt((acx - bcx)**2 + (acy - bcy)**2)
        return dist / 1000.0

    # ===============================
    # Weighted distance (KAIST 방식)
    # ===============================
    def weighted_distance(self, a, b, w):
        return (
            w["app"]  * self.d_app(a,b) +
            w["time"] * self.d_time(a,b) +
            w["cam"]  * self.d_cam(a,b) +
            w["pos"]  * self.d_pos(a,b)
        )

    # ===============================
    # Greedy association
    # ===============================
    def greedy(self, tracklets, threshold):
        N = len(tracklets)
        pairs = []

        # 모든 pair sim 계산
        for i in range(N):
            for j in range(i+1, N):
                if not self.temporal_compatible(tracklets[i], tracklets[j]):
                    continue
                sim = self.appearance_similarity(tracklets[i], tracklets[j])
                if sim >= threshold:
                    pairs.append((sim, i, j))

        pairs.sort(reverse=True, key=lambda x: x[0])
        assigned = [-1] * N
        gid = 0

        for sim, i, j in pairs:
            gi, gj = assigned[i], assigned[j]
            if gi == -1 and gj == -1:
                gid += 1
                assigned[i] = gid
                assigned[j] = gid
            elif gi != -1 and gj == -1:
                assigned[j] = gi
            elif gi == -1 and gj != -1:
                assigned[i] = gj

        # 미할당 → 단독 그룹
        for i in range(N):
            if assigned[i] == -1:
                gid += 1
                assigned[i] = gid

        # 그룹 결과 생성
        associations = []
        for g in range(1, gid+1):
            members = [tracklets[i] for i in range(N) if assigned[i]==g]
            associations.append({"global_id": g, "members": members})

        return associations

    # ===============================
    # Weight-based clustering
    # ===============================
    def cluster_with_weight(self, tracklets, weights, dist_thresh=1.0):
        N = len(tracklets)
        parent = list(range(N))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(N):
            for j in range(i+1, N):
                if not self.temporal_compatible(tracklets[i], tracklets[j]):
                    continue
                d = self.weighted_distance(tracklets[i], tracklets[j], weights)
                if d <= dist_thresh:
                    union(i,j)

        groups = {}
        for i in range(N):
            r = find(i)
            groups.setdefault(r, []).append(tracklets[i])

        associations = []
        gid = 0
        for _, members in groups.items():
            gid += 1
            associations.append({"global_id": gid, "members": members})
        return associations

    # ===============================
    # Public API
    # ===============================
    def associate(self, tracklets):
        if self.mode == "greedy":
            return self.greedy(tracklets, self.thresh)
        else:
            return self.cluster_with_weight(tracklets, self.weights)
