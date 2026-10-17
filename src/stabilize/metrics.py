"""Metric rung (jitter) và sai số cho overlay trên video.

Jitter: gia tốc rời rạc của điểm neo overlay (chân cầu thủ) SAU KHI bù chuyển động camera, chia cho
chiều cao box để không phụ thuộc người ở gần hay xa camera:

    a_t = p_t - 2 M_t p_{t-1} + M_t M_{t-1} p_{t-2}        (M_t: camera frame t-1 -> t)
    jitter = |a_t| / h_t * 100   (% chiều cao box mỗi frame^2)

Đưa 2 điểm cũ về toạ độ frame t (thay vì cộng dồn từ frame 0) để không bị trôi tích luỹ.
Người thật chạy mượt nên gia tốc nhỏ; rung của detector / bộ lọc làm a_t lớn.

Rung thấp không đủ: lọc càng mạnh càng mượt nhưng càng trễ so với người thật. Luôn báo cáo kèm sai số so
với GT (khoảng cách điểm neo tới điểm neo GT, cũng chia chiều cao box).
"""
import numpy as np

from src.stabilize.camera_motion import apply


def jitter_values(tracks: dict, motions: dict[int, np.ndarray], compensate: bool = True) -> np.ndarray:
    """tracks: id -> list (frame, x, y, h) theo thứ tự frame. motions: frame t -> M_t (t-1 -> t)."""
    vals = []
    eye = np.eye(3)
    for seq in tracks.values():
        for (f0, x0, y0, _), (f1, x1, y1, _), (f2, x2, y2, h2) in zip(seq, seq[1:], seq[2:]):
            if f2 - f0 != 2 or h2 <= 0:
                continue
            M2 = motions.get(f2, eye) if compensate else eye
            M1 = motions.get(f1, eye) if compensate else eye
            p1 = apply(M2, [x1, y1])[0]
            p0 = apply(M2 @ M1, [x0, y0])[0]
            vals.append(np.linalg.norm(np.array([x2, y2]) - 2 * p1 + p0) / h2 * 100)
    return np.array(vals)


def summarize_jitter(vals: np.ndarray) -> dict:
    if len(vals) == 0:
        return {"jitter_median": None, "jitter_p90": None, "n": 0}
    return {"jitter_median": round(float(np.median(vals)), 2), "jitter_p90": round(float(np.percentile(vals, 90)), 2), "n": int(len(vals))}
