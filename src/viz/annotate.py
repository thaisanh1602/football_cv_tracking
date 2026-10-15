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
    ) -> np.ndarray:
        scene = frame.copy()
        if len(people):
            # chưa phân đội thì tô màu theo class (player / goalkeeper / referee)
            lookup = people.class_id if teams is None else np.where(teams < 0, 3, teams)
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
