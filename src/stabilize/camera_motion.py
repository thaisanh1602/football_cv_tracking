"""Ước lượng chuyển động camera giữa 2 frame liên tiếp.

Camera broadcast lia / zoom liên tục, nên vị trí pixel của cầu thủ thay đổi cả khi họ đứng yên. Muốn đo
(và lọc) rung của overlay phải tách chuyển động camera ra trước. Cách làm: optical flow thưa (Lucas-Kanade)
trên các điểm góc của nền (cỏ, vạch, khán đài), bỏ vùng có người vì người tự di chuyển, rồi fit affine
từng phần (xoay + co giãn + tịnh tiến) bằng RANSAC.
"""
import cv2
import numpy as np

IDENTITY = np.eye(3)


class CameraMotion:
    """update(frame) -> ma trận 3x3 (affine) map toạ độ pixel frame trước -> frame hiện tại."""

    def __init__(self, scale: float = 0.5, max_corners: int = 400, ransac_px: float = 2.0):
        self.scale, self.max_corners, self.ransac_px = scale, max_corners, ransac_px
        self.prev_gray = None
        self.prev_mask = None

    def _mask(self, shape, exclude_xyxy) -> np.ndarray:
        mask = np.full(shape, 255, dtype=np.uint8)
        if exclude_xyxy is not None:
            for x1, y1, x2, y2 in np.asarray(exclude_xyxy) * self.scale:
                mask[max(int(y1), 0) : int(y2) + 1, max(int(x1), 0) : int(x2) + 1] = 0
        return mask

    def update(self, frame: np.ndarray, exclude_xyxy: np.ndarray | None = None) -> np.ndarray:
        """exclude_xyxy: box người trong frame hiện tại (pixel gốc) -> không lấy điểm trên người."""
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        mask = self._mask(gray.shape, exclude_xyxy)
        M = IDENTITY
        if self.prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(self.prev_gray, self.max_corners, 0.01, 8, mask=self.prev_mask)
            if pts is not None and len(pts) >= 10:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None)
                ok = status.ravel() == 1
                if ok.sum() >= 10:
                    A, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=self.ransac_px)
                    if A is not None:
                        # đổi từ toạ độ ảnh thu nhỏ về pixel gốc: chỉ phần tịnh tiến bị chia theo scale
                        A[:, 2] /= self.scale
                        M = np.vstack([A, [0, 0, 1]])
        self.prev_gray, self.prev_mask = gray, mask
        return M

    def reset(self):
        self.prev_gray = self.prev_mask = None


def apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return pts @ M[:2, :2].T + M[:2, 2]
