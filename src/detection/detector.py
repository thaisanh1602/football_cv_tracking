import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.utils.label_mapping import BALL_ID, CLASS_MAP


class Detector:
    """YOLO detector trả về sv.Detections với class_id theo 4 class chuẩn."""

    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.25, ball_conf: float = 0.1, device=None):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        # bóng dùng ngưỡng thấp hơn: trên SoccerNet hạ 0.25 -> 0.1 tăng recall bóng 0.485 -> 0.559,
        # false positive tăng thêm thì để BallTracker lọc
        self.ball_conf = ball_conf
        self.device = device
        # map class id của model -> class id chuẩn; class không có trong CLASS_MAP bị bỏ
        self.id_map = {i: CLASS_MAP[n] for i, n in self.model.names.items() if n in CLASS_MAP}
        if not self.id_map:
            raise ValueError(f"Không class nào của model khớp CLASS_MAP: {self.model.names}")

    def __call__(self, frame: np.ndarray, imgsz: int | None = None, conf: float | None = None) -> sv.Detections:
        result = self.model.predict(
            frame,
            imgsz=imgsz or self.imgsz,
            conf=conf if conf is not None else min(self.conf, self.ball_conf),
            device=self.device,
            verbose=False,
        )[0]
        dets = sv.Detections.from_ultralytics(result)
        keep = np.isin(dets.class_id, list(self.id_map))
        dets = dets[keep]
        dets.class_id = np.array([self.id_map[c] for c in dets.class_id], dtype=int)
        if conf is None:
            dets = dets[(dets.class_id == BALL_ID) | (dets.confidence >= self.conf)]
        return dets


def split_people_ball(dets: sv.Detections) -> tuple[sv.Detections, sv.Detections]:
    is_ball = dets.class_id == BALL_ID
    return dets[~is_ball], dets[is_ball]
