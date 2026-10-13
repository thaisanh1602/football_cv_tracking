from pathlib import Path

import supervision as sv


class MOTWriter:
    """Ghi kết quả tracking theo format MOTChallenge: frame,id,x,y,w,h,conf,-1,-1,-1 (frame bắt đầu từ 1)."""

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def write(self, frame_idx: int, dets: sv.Detections):
        if dets.tracker_id is None:
            return
        for (x1, y1, x2, y2), tid, conf in zip(dets.xyxy, dets.tracker_id, dets.confidence):
            self.f.write(f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.4f},-1,-1,-1\n")

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
