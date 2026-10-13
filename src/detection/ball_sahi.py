import numpy as np
import supervision as sv

from src.detection.detector import Detector
from src.utils.label_mapping import BALL_ID


class SlicedBallDetector:
    """Sliced inference (kiểu SAHI) chỉ cho quả bóng.

    Người vẫn detect full-frame; bóng nhỏ nên cắt frame thành tile rồi detect từng tile.
    """

    def __init__(
        self,
        detector: Detector,
        slice_wh: int = 640,
        overlap_wh: int = 128,
        tile_imgsz: int = 640,
        conf: float = 0.1,
    ):
        self.detector = detector
        self.tile_imgsz = tile_imgsz
        self.conf = conf
        self.slicer = sv.InferenceSlicer(
            callback=self._callback,
            slice_wh=slice_wh,
            overlap_wh=overlap_wh,
            iou_threshold=0.3,
        )

    def _callback(self, tile: np.ndarray) -> sv.Detections:
        dets = self.detector(tile, imgsz=self.tile_imgsz, conf=self.conf)
        return dets[dets.class_id == BALL_ID]

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        return self.slicer(frame)
