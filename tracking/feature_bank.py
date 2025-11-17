import torch
import torch.nn.functional as F

class TrackletFeatureExtractor:
    def extract(self, tracklets):
        """
        tracklet["feat"] = 256-dim aggregated embedding
        """
        for t in tracklets:
            embeds = torch.tensor(t["embeds"], dtype=torch.float32)  # (T,256)
            feat = embeds.mean(dim=0)
            feat = F.normalize(feat, dim=0)
            t["feat"] = feat
