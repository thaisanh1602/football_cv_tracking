"""Ghép 2 video cạnh nhau (trước / sau) kèm tiêu đề và số liệu, cho artifact so sánh chống rung.

    python scripts/side_by_side.py --left before.mp4 --right after.mp4 --output compare.mp4 \\
        --left-title "BEFORE" --left-note "raw homography, raw masks" \\
        --right-title "AFTER" --right-note "smoothed H + per-player Kalman + stabilized masks"
"""
import argparse

import cv2
import numpy as np
from tqdm import tqdm


def header(width: int, title: str, note: str, color) -> np.ndarray:
    band = np.full((70, width, 3), 25, np.uint8)
    cv2.putText(band, title, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(band, note, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--left-title", default="BEFORE")
    ap.add_argument("--right-title", default="AFTER")
    ap.add_argument("--left-note", default="")
    ap.add_argument("--right-note", default="")
    ap.add_argument("--width", type=int, default=1920, help="chiều rộng video ghép")
    args = ap.parse_args()

    caps = [cv2.VideoCapture(args.left), cv2.VideoCapture(args.right)]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25
    n = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps))
    half = args.width // 2
    heads = [header(half, args.left_title, args.left_note, (80, 80, 255)), header(half, args.right_title, args.right_note, (80, 220, 80))]

    writer = None
    for _ in tqdm(range(n), desc="Ghép"):
        frames = [c.read()[1] for c in caps]
        if any(f is None for f in frames):
            break
        cols = []
        for f, h in zip(frames, heads):
            f = cv2.resize(f, (half, int(f.shape[0] * half / f.shape[1])), interpolation=cv2.INTER_AREA)
            cols.append(np.vstack([h, f]))
        hh = max(c.shape[0] for c in cols)
        cols = [np.vstack([c, np.zeros((hh - c.shape[0], half, 3), np.uint8)]) for c in cols]
        out = np.hstack(cols)
        if writer is None:
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
        writer.write(out)
    if writer:
        writer.release()
    print(f"Đã ghi {args.output}")


if __name__ == "__main__":
    main()
