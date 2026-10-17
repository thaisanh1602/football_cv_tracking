import cv2
import numpy as np
import supervision as sv

# màu theo index: đội 0, đội 1, trọng tài, chưa rõ
TEAM_PALETTE = sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700", "#AAAAAA"])


class FootballAnnotator:
    def __init__(self):
        self.ellipse = sv.EllipseAnnotator(color=TEAM_PALETTE, thickness=2)
        self.label = sv.LabelAnnotator(
            color=TEAM_PALETTE,
            text_color=sv.Color.BLACK,
            text_position=sv.Position.BOTTOM_CENTER,
            text_scale=0.4,
        )
        self.triangle = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"), base=20, height=17, outline_thickness=1
        )

    def annotate(
        self,
        frame: np.ndarray,
        people: sv.Detections,
        ball: sv.Detections,
        teams: np.ndarray | None = None,
        speeds_kmh: np.ndarray | None = None,
        masks: list | None = None,
    ) -> np.ndarray:
        """masks: mask toàn khung (bool) cho từng người, None nếu không có -> tô bóng người bán trong suốt."""
        scene = frame.copy()
        if len(people):
            # chưa phân đội thì tô màu theo class (player / goalkeeper / referee)
            lookup = people.class_id if teams is None else np.where(teams < 0, 3, teams)
            if masks is not None:
                scene = fill_masks(scene, masks, lookup)
            labels = None
            if people.tracker_id is not None:
                labels = [f"#{tid}" for tid in people.tracker_id]
                if speeds_kmh is not None:
                    labels = [f"{lb} | {v:.0f} km/h" if np.isfinite(v) else lb for lb, v in zip(labels, speeds_kmh)]
            scene = self.ellipse.annotate(scene, people, custom_color_lookup=lookup)
            scene = self.label.annotate(scene, people, labels=labels, custom_color_lookup=lookup)
        if len(ball):
            scene = self.triangle.annotate(scene, ball)
        return scene


def fill_masks(scene: np.ndarray, masks: list, color_idx, alpha: float = 0.4) -> np.ndarray:
    overlay = scene.copy()
    for m, c in zip(masks, color_idx):
        if m is not None:
            overlay[m] = TEAM_PALETTE.by_idx(int(c)).as_bgr()
    return cv2.addWeighted(overlay, alpha, scene, 1 - alpha, 0)
