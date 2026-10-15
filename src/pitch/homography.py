"""Ước lượng homography ảnh -> sân (mét) từ keypoint, kiểm tra H xấu, và làm mượt theo thời gian."""
from dataclasses import dataclass

import cv2
import numpy as np

from src.pitch.config import PITCH, PitchConfig


@dataclass
class HomographyResult:
    H: np.ndarray  # 3x3, ảnh (px) -> sân (m)
    n_inliers: int
    reproj_err_px: float  # sai số trung bình trên các inlier, đo trên ảnh


def estimate_homography(
    img_pts: np.ndarray,
    conf: np.ndarray,
    pitch: PitchConfig = PITCH,
    min_conf: float = 0.5,
    min_points: int = 4,
    ransac_thresh_px: float = 10.0,
    max_reproj_err_px: float = 10.0,
    min_spread_m2: float = 50.0,
    img_wh: tuple[int, int] = (1920, 1080),
    max_outside_m: float = 30.0,
) -> HomographyResult | None:
    """Trả về None nếu không đủ điểm tin cậy, điểm gần thẳng hàng, hoặc H khớp kém.

    RANSAC chạy theo chiều sân -> ảnh với ngưỡng pixel, vì lỗi của model keypoint đo bằng pixel. Ngưỡng
    theo mét trên sân loại oan keypoint ở xa camera (1 px ở đó là hàng chục cm): trên SoccerNet GSR, đổi
    từ 1 m sang 10 px giảm sai số radar trung bình 20.5 -> 12.4 m.
    """
    keep = conf >= min_conf
    if keep.sum() < min_points:
        return None
    img, pit = img_pts[keep].astype(np.float32), pitch.vertices[keep].astype(np.float32)
    # các điểm sân phải trải ra một diện tích đủ lớn, không thì H suy biến (vd 4 điểm cùng 1 vạch)
    if cv2.contourArea(cv2.convexHull(pit)) < min_spread_m2:
        return None

    G, mask = cv2.findHomography(pit, img, cv2.RANSAC, ransac_thresh_px)  # sân -> ảnh
    if G is None or abs(np.linalg.det(G)) < 1e-12:
        return None
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < min_points:
        return None
    err = float(np.linalg.norm(cv2.perspectiveTransform(pit[inliers][None], G)[0] - img[inliers], axis=1).mean())
    if err > max_reproj_err_px:
        return None
    H = np.linalg.inv(G)
    if not plausible(H, img_wh, pitch, max_outside_m):
        return None
    return HomographyResult(H=H, n_inliers=int(inliers.sum()), reproj_err_px=err)


def image_grid(img_wh: tuple[int, int] = (1920, 1080), nx: int = 5, ny: int = 3) -> np.ndarray:
    """Lưới điểm cố định ở 2/3 dưới khung hình: với camera broadcast gần như luôn là mặt cỏ."""
    w, h = img_wh
    gx, gy = np.meshgrid(np.linspace(0.05 * w, 0.95 * w, nx), np.linspace(0.4 * h, 0.95 * h, ny))
    return np.column_stack([gx.ravel(), gy.ravel()])


def plausible(H: np.ndarray, img_wh, pitch: PitchConfig = PITCH, max_outside_m: float = 30.0) -> bool:
    """Loại H "thảm hoạ" mà sai số chiếu lại không bắt được (vd 4 keypoint gần nhau, H ngoại suy sai xa):
    vùng cỏ của ảnh phải chiếu vào trong sân (cho phép lệch max_outside_m), và không đi qua đường chân trời."""
    pts = np.c_[image_grid(img_wh), np.ones(15)] @ H.T
    if np.any(pts[:, 2] <= 0) or np.sign(pts[:, 2]).min() != np.sign(pts[:, 2]).max():
        return False
    xy = pts[:, :2] / pts[:, 2:]
    m = max_outside_m
    return bool(np.all((xy[:, 0] > -m) & (xy[:, 0] < pitch.length + m) & (xy[:, 1] > -m) & (xy[:, 1] < pitch.width + m)))


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.empty((0, 2))
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


class OneEuroFilter:
    """One-Euro filter (Casiez 2012) cho mảng: mượt khi tín hiệu đứng yên, bám nhanh khi tín hiệu đổi nhanh."""

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * self.freq)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x, np.zeros_like(x)
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

    def reset(self):
        self.x_prev = self.dx_prev = None


class HomographySmoother:
    """Làm mượt H theo thời gian.

    Không lọc trực tiếp 9 hệ số của H (khác đơn vị, khác độ lớn). Thay vào đó lấy một lưới điểm cố định
    trên ẢNH (vùng cỏ), chiếu ra sân bằng H, lọc toạ độ sân (m) bằng One-Euro, rồi fit lại H.
    Neo trên ảnh chứ không trên sân: điểm sân nằm ngoài khung hình chiếu vào ảnh rất bất ổn (gần chân trời
    toạ độ tiến ra vô cùng), lọc chúng làm H tệ hơn. Trên GSR, bản neo trên sân làm rung tăng gấp 2-4 lần.
    - frame không có H tốt: giữ H gần nhất tối đa `max_hold` frame
    - lưới nhảy quá `cut_m` (cắt cảnh / đổi camera): reset bộ lọc thay vì trượt dần sang góc mới
    """

    def __init__(
        self,
        fps: float = 25,
        min_cutoff: float = 1.0,
        beta: float = 0.02,
        max_hold: int = 25,
        cut_m: float = 10.0,
        img_wh: tuple[int, int] = (1920, 1080),
    ):
        self.anchors = image_grid(img_wh)
        self.filter = OneEuroFilter(fps, min_cutoff=min_cutoff, beta=beta)
        self.max_hold, self.cut_m = max_hold, cut_m
        self.H = None
        self.missed = 0
        self.prev_m = None

    def update(self, result: HomographyResult | None) -> np.ndarray | None:
        if result is None:
            self.missed += 1
            if self.missed > self.max_hold:
                self.H, self.prev_m = None, None
                self.filter.reset()
            return self.H
        self.missed = 0

        pts_m = project(result.H, self.anchors)
        if self.prev_m is not None and np.median(np.linalg.norm(pts_m - self.prev_m, axis=1)) > self.cut_m:
            self.filter.reset()
        self.prev_m = pts_m
        smoothed = self.filter(pts_m)
        self.H, _ = cv2.findHomography(self.anchors.astype(np.float32), smoothed.astype(np.float32), 0)
        return self.H


class PitchMapper:
    """Gộp: keypoint -> H thô -> (làm mượt) -> chiếu điểm ảnh ra toạ độ sân (m)."""

    def __init__(self, keypoint_detector, smooth: bool = True, fps: float = 25):
        self.detect = keypoint_detector
        self.smoother = HomographySmoother(fps=fps) if smooth else None
        self.H = None
        self.last_raw: HomographyResult | None = None

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        self.last_raw = estimate_homography(*self.detect(frame))
        if self.smoother is not None:
            self.H = self.smoother.update(self.last_raw)
        else:
            self.H = self.last_raw.H if self.last_raw else None
        return self.H

    def to_pitch(self, img_pts: np.ndarray) -> np.ndarray | None:
        return None if self.H is None else project(self.H, img_pts)
