import cv2
import numpy as np

from src.pitch.config import PITCH, PitchConfig

# BGR, cùng thứ tự màu với src/viz/annotate.py: đội 0, đội 1, trọng tài, chưa rõ
TEAM_COLORS_BGR = [(255, 191, 0), (147, 20, 255), (0, 215, 255), (170, 170, 170)]
BALL_COLOR_BGR = (255, 255, 255)
GRASS_BGR = (34, 139, 34)


class Radar:
    """Vẽ sân 2D nhìn từ trên xuống, toạ độ mét -> pixel theo `scale` px/m."""

    def __init__(self, scale: float = 8.0, padding: int = 30, pitch: PitchConfig = PITCH):
        self.scale, self.padding, self.pitch = scale, padding, pitch
        self.base = self._draw_pitch()

    def to_px(self, xy_m: np.ndarray) -> np.ndarray:
        # kẹp giá trị: H gần suy biến có thể chiếu ra toạ độ cực lớn làm tràn int khi vẽ
        return np.clip(np.asarray(xy_m) * self.scale + self.padding, -10_000, 10_000).astype(int)

    def _draw_pitch(self) -> np.ndarray:
        p, s = self.pitch, self.scale
        h, w = int(p.width * s) + 2 * self.padding, int(p.length * s) + 2 * self.padding
        img = np.full((h, w, 3), GRASS_BGR, dtype=np.uint8)
        v = self.to_px(p.vertices)
        for a, b in p.edges:
            cv2.line(img, tuple(v[a]), tuple(v[b]), (255, 255, 255), 2, cv2.LINE_AA)
        centre = tuple(self.to_px([p.length / 2, p.width / 2]))
        cv2.circle(img, centre, int(p.centre_circle_radius * s), (255, 255, 255), 2, cv2.LINE_AA)
        for spot in (v[8], v[21]):
            cv2.circle(img, tuple(spot), 3, (255, 255, 255), -1, cv2.LINE_AA)
        return img

    def draw(
        self,
        xy_m: np.ndarray,
        teams: np.ndarray | None = None,
        ball_m: np.ndarray | None = None,
        trails: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """trails: với mỗi người, các vị trí gần đây (m) theo thứ tự thời gian -> vẽ vệt mờ dần."""
        img = self.base.copy()
        teams = np.full(len(xy_m), 3) if teams is None else np.where(teams < 0, 3, teams)
        if trails is not None:
            for trail, t in zip(trails, teams):
                if len(trail) < 2:
                    continue
                px = self.to_px(trail)
                for k in range(1, len(px)):
                    a = k / len(px)  # cũ -> hoà vào màu cỏ, mới -> màu đội
                    color = tuple(int(c * a + g * (1 - a)) for c, g in zip(TEAM_COLORS_BGR[t], GRASS_BGR))
                    cv2.line(img, tuple(px[k - 1]), tuple(px[k]), color, 2, cv2.LINE_AA)
        for pt, t in zip(self.to_px(xy_m), teams):
            cv2.circle(img, tuple(pt), 8, TEAM_COLORS_BGR[t], -1, cv2.LINE_AA)
            cv2.circle(img, tuple(pt), 8, (0, 0, 0), 1, cv2.LINE_AA)
        if ball_m is not None and len(ball_m):
            for pt in self.to_px(ball_m):
                cv2.circle(img, tuple(pt), 5, BALL_COLOR_BGR, -1, cv2.LINE_AA)
                cv2.circle(img, tuple(pt), 5, (0, 0, 0), 1, cv2.LINE_AA)
        return img


    def empty(self, message: str = "Khong du vach san de dung radar") -> np.ndarray:
        img = (self.base * 0.6).astype(np.uint8)
        cv2.putText(img, message, (self.padding + 20, img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return img


def overlay_radar(frame: np.ndarray, radar_img: np.ndarray, width_ratio: float = 0.4, alpha: float = 0.85) -> np.ndarray:
    """Chồng radar vào giữa phía dưới khung hình."""
    h, w = frame.shape[:2]
    rw = int(w * width_ratio)
    rh = int(radar_img.shape[0] * rw / radar_img.shape[1])
    small = cv2.resize(radar_img, (rw, rh), interpolation=cv2.INTER_AREA)
    x0, y0 = (w - rw) // 2, h - rh - 10
    roi = frame[y0 : y0 + rh, x0 : x0 + rw]
    frame[y0 : y0 + rh, x0 : x0 + rw] = cv2.addWeighted(small, alpha, roi, 1 - alpha, 0)
    return frame
