import numpy as np
import supervision as sv

from src.detection.detector import Detector
from src.utils.label_mapping import BALL_ID


class SlicedBallDetector:
    """Sliced inference (kiểu SAHI) chỉ cho quả bóng.

    Người vẫn detect full-frame; bóng nhỏ nên cắt frame thành tile rồi detect từng tile.
    Giữ tile_imgsz / slice_wh ~ imgsz lúc train / 1920 (1280/1920 ~ 640/960) để bóng trong tile
    cùng kích thước như lúc train. Tile 640 -> 640 làm bóng to gấp 1.5 lần và recall giảm 0.549 -> 0.42.
    """

    def __init__(
        self,
        detector: Detector,
        slice_wh: int = 960,
        overlap_wh: int = 192,
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
