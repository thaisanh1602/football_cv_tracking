"""Xuất annotation dạng JSON cho web: trình duyệt phát video gốc và tự vẽ overlay, server không stream video đã render.

Schema (version 1):
{
  "version": 1,
  "video": {"fps", "width", "height", "frames"},
  "pitch": {"length", "width", "vertices": [[x, y]...], "edges": [[i, j]...], "centre_circle_radius"},   # mét
  "teams": ["#00BFFF", "#FF1493", "#FFD700", "#AAAAAA"],      # màu theo index: đội 0, đội 1, trọng tài, chưa rõ
  "frames": [
    {"f": 0,                                                   # chỉ số frame, 0-based
     "h": true,                                                # frame này có homography (radar dùng được) không
     "p": [{"id": 7, "t": 0, "c": 0,                           # track id, đội (index màu), class (0 player, 1 GK, 2 ref)
            "b": [x1, y1, x2, y2],                             # box pixel
            "v": 18.4,                                         # tốc độ km/h (null nếu chưa đủ dữ liệu)
            "r": [x, y], "k": [x, y]}],                        # vị trí sân (m): thô / đã lọc Kalman (null nếu không có H)
     "ball": {"b": [x1, y1, x2, y2], "r": [x, y]} | null}
  ]
}
Khoá viết tắt và làm tròn 1 chữ số thập phân để file 30 s (~750 frame x 25 người) chỉ cỡ vài MB.
"""
import json
from pathlib import Path

import numpy as np

from src.pitch.config import PITCH
from src.utils.label_mapping import REFEREE_ID

TEAM_COLORS = ["#00BFFF", "#FF1493", "#FFD700", "#AAAAAA"]


def _r(a) -> list | None:
    return None if a is None else [round(float(x), 1) for x in np.asarray(a).ravel()]


class AnnotationWriter:
    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        self.path = Path(path)
        self.meta = {"fps": fps, "width": width, "height": height}
        self.frames = []

    def add(self, f: int, people, teams=None, speeds=None, pitch_raw=None, pitch_kf=None, ball=None, ball_m=None):
        players = []
        for k in range(len(people)):
            cls = int(people.class_id[k])
            team = int(teams[k]) if teams is not None else (2 if cls == REFEREE_ID else 3)
            v = None if speeds is None or not np.isfinite(speeds[k]) else round(float(speeds[k]), 1)
            players.append({
                "id": int(people.tracker_id[k]),
                "t": 3 if team < 0 else team,
                "c": cls,
                "b": _r(people.xyxy[k]),
                "v": v,
                "r": None if pitch_raw is None else _r(pitch_raw[k]),
                "k": None if pitch_kf is None else _r(pitch_kf[k]),
            })
        b = None
        if ball is not None and len(ball):
            b = {"b": _r(ball.xyxy[0]), "r": None if ball_m is None else _r(ball_m[0])}
        self.frames.append({"f": f, "h": pitch_raw is not None, "p": players, "ball": b})

    def close(self):
        doc = {
            "version": 1,
            "video": self.meta | {"frames": len(self.frames)},
            "pitch": {
                "length": PITCH.length,
                "width": PITCH.width,
                "vertices": [_r(v) for v in PITCH.vertices],
                "edges": [list(map(int, e)) for e in PITCH.edges],
                "centre_circle_radius": PITCH.centre_circle_radius,
            },
            "teams": TEAM_COLORS,
            "frames": self.frames,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
