import numpy as np
import supervision as sv
import yaml
from ultralytics.engine.results import Boxes
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace


class PeopleTracker:
    """Bọc tracker của Ultralytics (ByteTrack / BoT-SORT / BoT-SORT+ReID) để nhận sv.Detections.

    Dùng trực tiếp class tracker thay vì model.track() để tách detect và track,
    nhờ vậy có thể ghép thêm SAHI cho bóng, phân đội, xuất MOT...
    """

    def __init__(self, cfg_path: str, device=None):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = IterableSimpleNamespace(**yaml.safe_load(f))
        cfg.device = device
        if getattr(cfg, "with_reid", False) and cfg.model == "auto":
            raise ValueError("model: auto chỉ dùng được trong model.track(); hãy chỉ định file ReID, vd yolo26n-cls.pt")
        self.tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)

    def update(self, dets: sv.Detections, frame: np.ndarray) -> sv.Detections:
        # luôn gọi update kể cả khi không có detection để tracker đếm frame bị mất
        data = np.column_stack([dets.xyxy, dets.confidence, dets.class_id]).astype(np.float32)
        tracks = self.tracker.update(Boxes(data.reshape(-1, 6), frame.shape[:2]), frame)
        if len(tracks) == 0:
            return sv.Detections.empty()
        # mỗi track: x1, y1, x2, y2, id, score, cls, idx (idx = vị trí trong dets)
        out = dets[tracks[:, 7].astype(int)]
        out.xyxy = tracks[:, :4]
        out.tracker_id = tracks[:, 4].astype(int)
        return out

    def reset(self):
        self.tracker.reset()
