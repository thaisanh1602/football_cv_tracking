import numpy as np
from ultralytics import YOLO


class PitchKeypointDetector:
    """YOLO-pose phát hiện 32 keypoint sân. Trả về toạ độ ảnh (32, 2) và độ tin cậy (32,)."""

    def __init__(self, weights: str, imgsz: int = 1280, device=None):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.device = device

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = self.model.predict(frame, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return np.zeros((32, 2)), np.zeros(32)
        best = int(result.boxes.conf.argmax())  # cả sân là 1 "object"; lấy object tin cậy nhất
        xy = result.keypoints.xy[best].cpu().numpy()
        conf = result.keypoints.conf[best].cpu().numpy() if result.keypoints.conf is not None else np.ones(len(xy))
        return xy, conf
