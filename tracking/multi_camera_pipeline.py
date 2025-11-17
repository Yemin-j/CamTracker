from .tracklet_builder import TrackletBuilder
from .feature_bank import TrackletFeatureExtractor
from .mcta import MultiCameraAssociator
from .global_id_mapper import GlobalIDMapper

class MultiCameraTrackingPipeline:

    def __init__(self, mode="greedy", weights=None, thresh=0.5):
        self.builder = TrackletBuilder()
        self.extractor = TrackletFeatureExtractor()
        self.associator = MultiCameraAssociator(
            mode=mode, weights=weights, thresh=thresh
        )
        self.mapper = GlobalIDMapper()

    def run(self, all_tracks):
        # 1) 프레임 기반 기록 → 트랙 단위 tracklet 생성
        tracklets = self.builder.build(all_tracks)

        # 2) embedding 평균내기
        self.extractor.extract(tracklets)

        # 3) multi-camera association
        associations = self.associator.associate(tracklets)

        # 4) global_id 매핑 생성
        mapping = self.mapper.build_mapping(associations)

        # 5) frame-level 기록에 global_id 적용
        final_tracks = self.mapper.apply(all_tracks, mapping)

        return final_tracks, associations
