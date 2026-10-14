"""Phân loại nguyên nhân ID switch từ file kết quả MOT của eval_soccernet.py.

Mỗi frame ghép GT <-> track bằng Hungarian (IoU >= 0.5), ưu tiên giữ cặp đã ghép ở frame trước như TrackEval.
Khi một người GT đổi sang track ID khác:
- "tráo ID với người khác": ID mới đã từng thuộc về người GT khác -> hai người gần nhau bị gán nhầm
- "2 track cùng bám 1 người": track cũ vẫn còn trong frame này -> tracker sinh track thừa rồi lật qua lại
- "track mới sau khi mất dấu": có khoảng frame không ghép được rồi ra ID mới
  -> tracker không nối lại được (detector bỏ sót, track_buffer ngắn)
- "track cũ đứt, track mới thay ngay": frame trước còn ghép đúng, frame này track cũ biến mất và ID mới xuất hiện
  -> dự đoán Kalman lệch khỏi detection (camera lia, người đổi hướng) nên tracker tạo track mới
Kèm tỷ lệ switch xảy ra khi người đó đang chồng lấn với người khác (che khuất).

    python scripts/analyze_idsw.py --data-root <SoccerNet split> --results outputs/trackeval/bytetrack/data
"""
import argparse
import configparser
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_mot(path: Path, exclude_ids=()) -> dict[int, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    data = data[~np.isin(data[:, 1], list(exclude_ids))]
    return {int(f): data[data[:, 0] == f][:, 1:6] for f in np.unique(data[:, 0])}  # id, x, y, w, h


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.column_stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]])
    b = np.column_stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def ball_ids(seq_dir: Path) -> list[int]:
    info = configparser.ConfigParser()
    info.read(seq_dir / "gameinfo.ini")
    return [int(k.split("_")[1]) for k, v in info["Sequence"].items() if k.startswith("trackletid_") and v.strip().startswith("ball")]


def analyze(seq_dir: Path, result_file: Path, counter: Counter, gaps: list):
    gt = load_mot(seq_dir / "gt" / "gt.txt", exclude_ids=ball_ids(seq_dir))  # chỉ đánh giá người
    pred = load_mot(result_file)
    last = {}  # gt id -> (track id, frame cuối ghép được)
    owner = {}  # track id -> gt id đã từng ghép
    for f in sorted(gt):
        g, p = gt[f], pred.get(f, np.empty((0, 5)))
        if len(p) == 0:
            continue
        iou = iou_matrix(g[:, 1:], p[:, 1:])
        keep = np.array([[last.get(int(gi), (None,))[0] == int(ti) for ti in p[:, 0]] for gi in g[:, 0]])
        rows, cols = linear_sum_assignment(-(iou + 1000 * (keep & (iou >= 0.5))))
        gt_overlap = iou_matrix(g[:, 1:], g[:, 1:])
        np.fill_diagonal(gt_overlap, 0)
        for r, c in zip(rows, cols):
            if iou[r, c] < 0.5:
                continue
            gid, tid = int(g[r, 0]), int(p[c, 0])
            if gid in last and last[gid][0] != tid:
                gap = f - last[gid][1] - 1
                if tid in owner and owner[tid] != gid:
                    kind = "tráo ID với người khác"
                elif last[gid][0] in p[:, 0]:
                    kind = "2 track cùng bám 1 người"
                elif gap > 0:
                    kind = "track mới sau khi mất dấu"
                    gaps.append(gap)
                else:
                    kind = "track cũ đứt, track mới thay ngay"
                counter[kind] += 1
                counter["(đang chồng lấn người khác)"] += int(gt_overlap[r].max() > 0.1)
                counter["tổng"] += 1
            last[gid] = (tid, f)
            owner.setdefault(tid, gid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--results", nargs="+", required=True, help="một hoặc nhiều thư mục <tracker>/data")
    args = ap.parse_args()

    for res_dir in map(Path, args.results):
        counter, gaps = Counter(), []
        for f in sorted(res_dir.glob("*.txt")):
            analyze(Path(args.data_root) / f.stem, f, counter, gaps)
        total = counter.pop("tổng", 0)
        print(f"\n## {res_dir.parent.name}: {total} switch (IoU>=0.5)")
        for kind, n in counter.most_common():
            print(f"- {kind}: {n} ({n / max(total, 1):.0%})")
        if gaps:
            print(f"- gap trước khi ra track mới: trung vị {int(np.median(gaps))} frame, 90% <= {int(np.percentile(gaps, 90))} frame")


if __name__ == "__main__":
    main()
