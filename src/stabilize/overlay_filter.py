"""Lọc vị trí overlay theo từng track, chạy nhân quả (chỉ dùng frame hiện tại và trước đó) để vẽ được live.

Dùng cho toạ độ sân (m) trên radar: homography nhảy vài pixel giữa 2 frame thành vài chục cm trên sân, nên
chấm radar rung gấp ~20 lần người thật (đo trên SoccerNet GSR). Lọc đặt SAU homography, theo từng cầu thủ.

- KalmanCV: Kalman vận tốc không đổi, trạng thái (x, y, vx, vy). Mất vài frame thì dự đoán tiếp theo vận tốc.
  `accel_std` (m/s^2) là độ "linh hoạt": nhỏ -> mượt nhưng trễ khi đổi hướng; lớn -> bám nhanh nhưng rung.
- OneEuro2D: One-Euro theo từng trục (mượt khi đứng yên, bám nhanh khi chạy).
- EMA: trung bình trượt mũ, mốc so sánh đơn giản nhất.
TrackFilterBank giữ 1 bộ lọc cho mỗi track id, reset khi track "dịch chuyển tức thời" (ID switch, H hỏng).
"""
import numpy as np

from src.pitch.homography import OneEuroFilter
from src.pitch.kinematics import JUMP_MARGIN_M, MAX_SPEED_MS


class KalmanCV:
    def __init__(self, fps: float = 25, accel_std: float = 2.0, meas_std: float = 1.0):
        self.dt = 1.0 / fps
        dt = self.dt
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        # nhiễu quá trình từ gia tốc ngẫu nhiên (mô hình white-noise acceleration)
        g = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        self.Q = g @ g.T * accel_std**2
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2) * meas_std**2
        self.x = None
        self.P = None

    def predict(self, steps: int = 1):
        for _ in range(steps):
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray, steps: int = 1) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([self.R[0, 0], self.R[1, 1], 25.0, 25.0])  # vận tốc ban đầu chưa biết (~5 m/s)
            return z
        self.predict(steps)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].copy()


class OneEuro2D:
    def __init__(self, fps: float = 25, min_cutoff: float = 0.5, beta: float = 0.1):
        self.f = OneEuroFilter(fps, min_cutoff=min_cutoff, beta=beta)

    def update(self, z, steps: int = 1) -> np.ndarray:
        return self.f(np.asarray(z, dtype=float))


class EMA:
    def __init__(self, alpha: float = 0.3):
        self.alpha, self.x = alpha, None

    def update(self, z, steps: int = 1) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        self.x = z if self.x is None else self.alpha * z + (1 - self.alpha) * self.x
        return self.x.copy()


class TrackFilterBank:
    """make_filter(): tạo bộ lọc mới cho 1 track. update() nhận id + vị trí của các track trong frame."""

    def __init__(self, make_filter, fps: float = 25, max_gap: int = 12):
        self.make, self.fps, self.max_gap = make_filter, fps, max_gap
        self.filters: dict[int, object] = {}
        self.last: dict[int, tuple[int, np.ndarray]] = {}

    def update(self, frame_idx: int, track_ids, xy) -> np.ndarray:
        out = np.asarray(xy, dtype=float).copy().reshape(-1, 2)
        for k, (tid, z) in enumerate(zip(track_ids, out.copy())):
            steps = 1
            if tid in self.last:
                f0, z0 = self.last[tid]
                steps = frame_idx - f0
                if steps > self.max_gap or np.linalg.norm(z - z0) > MAX_SPEED_MS * steps / self.fps + JUMP_MARGIN_M:
                    self.filters.pop(tid, None)
                    steps = 1
            filt = self.filters.setdefault(tid, self.make())
            out[k] = filt.update(z, steps=max(steps, 1))
            self.last[tid] = (frame_idx, z)
        return out
