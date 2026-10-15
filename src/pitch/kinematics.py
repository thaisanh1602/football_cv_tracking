"""Tốc độ và quãng đường từ quỹ đạo trên sân (mét).

Không tính tốc độ từ 2 frame liên tiếp: nhiễu vị trí 10 cm giữa 2 frame cách nhau 0.04 s đã thành 9 km/h.
Thay vào đó làm mượt quỹ đạo bằng Savitzky-Golay (đa thức bậc 2 trên cửa sổ 2 s) và lấy đạo hàm của đa thức.
Quỹ đạo được cắt thành các đoạn liền tại chỗ mất dấu lâu hoặc "dịch chuyển tức thời" (ID switch, H hỏng),
không nối qua các chỗ đó, nếu không 1 cú nhảy 20 m sẽ cộng thẳng vào quãng đường.
"""
from collections import deque

import numpy as np
from scipy.signal import savgol_filter

MAX_SPEED_MS = 12.0  # ~43 km/h, nhanh hơn kỷ lục chạy nước rút (~10.4 m/s): vượt mức này chắc chắn là lỗi
JUMP_MARGIN_M = 2.0  # cho phép thêm vì vị trí từng frame có nhiễu


def split_segments(frames: np.ndarray, xy: np.ndarray, fps: float = 25, max_gap: int = 12) -> list[slice]:
    """Cắt tại gap > max_gap frame hoặc bước nhảy nhanh hơn MAX_SPEED_MS (cộng biên nhiễu)."""
    cuts = [0]
    for i in range(1, len(frames)):
        df = frames[i] - frames[i - 1]
        if df > max_gap or np.linalg.norm(xy[i] - xy[i - 1]) > MAX_SPEED_MS * df / fps + JUMP_MARGIN_M:
            cuts.append(i)
    cuts.append(len(frames))
    return [slice(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def smooth_segment(frames: np.ndarray, xy: np.ndarray, fps: float = 25, window_s: float = 2.0):
    """Một đoạn liền -> (frame đều, vị trí đã làm mượt, tốc độ m/s). Gap nhỏ được nội suy tuyến tính.
    Đoạn ngắn hơn cửa sổ trả về tốc độ None (không đủ dữ liệu để ước lượng tin cậy)."""
    f = np.arange(frames[0], frames[-1] + 1)
    x, y = np.interp(f, frames, xy[:, 0]), np.interp(f, frames, xy[:, 1])
    win = int(round(window_s * fps)) | 1  # số lẻ
    if len(f) < win:
        return f, np.column_stack([x, y]), None
    pos = np.column_stack([savgol_filter(x, win, 2), savgol_filter(y, win, 2)])
    vel = np.column_stack([savgol_filter(x, win, 2, deriv=1), savgol_filter(y, win, 2, deriv=1)]) * fps
    return f, pos, np.linalg.norm(vel, axis=1)


def track_kinematics(frames, xy, fps: float = 25, window_s: float = 2.0) -> dict:
    """Quỹ đạo 1 track (frame tăng dần) -> {frame: tốc độ m/s}, tổng quãng đường (m), số đoạn bị cắt."""
    frames, xy = np.asarray(frames), np.asarray(xy, dtype=float)
    speed, distance = {}, 0.0
    segments = split_segments(frames, xy, fps)
    for seg in segments:
        f, pos, v = smooth_segment(frames[seg], xy[seg], fps, window_s)
        if v is None:
            continue
        distance += float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
        speed.update(zip(f.tolist(), v.tolist()))
    return {"speed": speed, "distance": distance, "segments": len(segments)}


class OnlineKinematics:
    """Bản chạy theo từng frame cho video: với mỗi track, fit đường thẳng (vị trí theo thời gian) trên
    `window_s` giây gần nhất -> vận tốc = hệ số góc, vị trí hiện tại = giá trị fit. Quãng đường cộng dồn
    theo vị trí đã fit; bước nhảy bất thường thì xoá lịch sử (coi như đoạn mới) thay vì cộng vào."""

    def __init__(self, fps: float = 25, window_s: float = 2.0):
        self.fps, self.n = fps, max(int(window_s * fps), 3)
        self.hist: dict[int, deque] = {}
        self.last_fit: dict[int, tuple[int, np.ndarray]] = {}
        self.distance: dict[int, float] = {}

    def update(self, frame_idx: int, track_ids, xy_m) -> np.ndarray:
        """Trả về tốc độ km/h cho từng track (nan khi chưa đủ dữ liệu)."""
        speeds = np.full(len(track_ids), np.nan)
        for k, (tid, p) in enumerate(zip(track_ids, np.asarray(xy_m, dtype=float))):
            h = self.hist.setdefault(tid, deque(maxlen=self.n))
            if h:
                f0, p0 = h[-1]
                df = frame_idx - f0
                if df > 12 or np.linalg.norm(p - p0) > MAX_SPEED_MS * df / self.fps + JUMP_MARGIN_M:
                    h.clear()
                    self.last_fit.pop(tid, None)
            h.append((frame_idx, p))
            if len(h) < self.n // 2:
                continue
            t = np.array([f for f, _ in h], dtype=float) / self.fps
            P = np.array([q for _, q in h])
            A = np.column_stack([t - t[-1], np.ones(len(t))])
            coef, *_ = np.linalg.lstsq(A, P, rcond=None)  # hàng 0: vận tốc, hàng 1: vị trí hiện tại
            speeds[k] = np.linalg.norm(coef[0]) * 3.6
            if tid in self.last_fit:
                self.distance[tid] = self.distance.get(tid, 0.0) + float(np.linalg.norm(coef[1] - self.last_fit[tid][1]))
            self.last_fit[tid] = (frame_idx, coef[1])
        return speeds
