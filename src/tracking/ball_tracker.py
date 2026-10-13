import numpy as np
import supervision as sv

BALL_TRACK_ID = 10_000  # id cố định cho bóng, tránh trùng id cầu thủ khi xuất MOT


class BallTracker:
    """Giữ tối đa 1 quả bóng mỗi frame và loại các điểm nhảy bất thường.

    Không đưa bóng vào ByteTrack: bóng nhỏ, nhanh, hay bị che nên IoU matching không hợp.
    """

    def __init__(self, max_jump_px: float = 120, max_lost: int = 25):
        self.max_jump_px = max_jump_px
        self.max_lost = max_lost
        self.last_center = None
        self.lost = 0

    def update(self, ball: sv.Detections) -> sv.Detections:
        if len(ball) == 0:
            return self._miss()

        centers = ball.get_anchors_coordinates(sv.Position.CENTER)
        if self.last_center is None:
            i = int(np.argmax(ball.confidence))
        else:
            dist = np.linalg.norm(centers - self.last_center, axis=1)
            i = int(np.argmin(dist))
            # cho phép nhảy xa hơn nếu đã mất bóng nhiều frame
            if dist[i] > self.max_jump_px * (self.lost + 1):
                return self._miss()

        self.last_center = centers[i]
        self.lost = 0
        out = ball[[i]]
        out.tracker_id = np.array([BALL_TRACK_ID])
        return out

    def _miss(self) -> sv.Detections:
        self.lost += 1
        if self.lost > self.max_lost:
            self.last_center = None
        return sv.Detections.empty()
