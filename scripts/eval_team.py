"""Đo độ chính xác phân đội trên SoccerNet-Tracking (nhãn đội lấy từ gameinfo.ini: "player team left/right").

So sánh: KMeans màu áo vs SigLIP + PCA vs SigLIP + UMAP, KMeans 2 cụm vs 3 cụm, mỗi cách đo
- theo từng crop (phân loại mỗi frame độc lập)
- theo bỏ phiếu track ID (TeamAssigner, đúng code pipeline infer_video): N phiếu đầu vs cửa sổ trượt
- thủ môn: gán theo đội có tâm vị trí gần nhất (chỉ tính thủ môn được detector nhận đúng class)
KMeans cho nhãn 0/1 tùy ý nên mỗi sequence lấy hoán vị 0/1 <-> trái/phải tốt nhất.

    python scripts/eval_team.py --data-root <SoccerNet split> --weights outputs/best.pt --limit 10
"""
import argparse
import configparser
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_idsw import iou_matrix  # noqa: E402
from scripts.eval_soccernet import detect_sequence, list_frames  # noqa: E402
from src.detection.detector import Detector  # noqa: E402
from src.team.team_classifier import ColorEmbedder, SiglipEmbedder, TeamAssigner, TeamClassifier, torso_crops  # noqa: E402
from src.tracking.trackers import PeopleTracker  # noqa: E402
from src.utils.label_mapping import BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID  # noqa: E402

METHODS = [("color", None), ("siglip", "pca"), ("siglip", "umap")]
CLUSTERS = [2, 3]
VOTING = {"10 đầu": (10, False), "trượt 5": (5, True), "trượt 15": (15, True), "trượt 30": (30, True)}
DEFAULT_VOTING = "trượt 15"


def load_gt(seq_dir: Path):
    """frame -> mảng (id, x, y, w, h) và id -> (vai trò, "left"/"right"/None)."""
    info = configparser.ConfigParser()
    info.read(seq_dir / "gameinfo.ini")
    roles = {}
    for k, v in info["Sequence"].items():
        if k.startswith("trackletid_"):
            label = v.split(";")[0].strip()
            side = "left" if "left" in label else "right" if "right" in label else None
            role = "player" if label.startswith("player") else "goalkeeper" if label.startswith("goalkeeper") else label
            roles[int(k.split("_")[1])] = (role, side)
    gt = np.loadtxt(seq_dir / "gt" / "gt.txt", delimiter=",", ndmin=2)
    return {int(f): gt[gt[:, 0] == f][:, 1:6] for f in np.unique(gt[:, 0])}, roles


def collect(seq_dir: Path, detector, cache_dir: Path, tracker_cfg, stride: int, fit_stride: int, device):
    """Chạy tracker trên cả sequence; ở các frame lấy mẫu, ghép track với GT và cắt crop áo cầu thủ."""
    frames = list_frames(seq_dir)
    dets_all = detect_sequence(frames, cache_dir / f"{seq_dir.name}.npz", detector, None)
    gt_by_frame, roles = load_gt(seq_dir)
    tracker = PeopleTracker(tracker_cfg, device=device)

    crops, fit_idx, frames_out = [], [], []
    for i, (path, dets) in enumerate(zip(frames, dets_all)):
        frame = cv2.imread(str(path))
        people = tracker.update(dets[dets.class_id != BALL_ID], frame)
        if i % stride or len(people) == 0:
            continue
        # ghép track <-> GT để biết đội thật của từng track ở frame này
        gt_side = [None] * len(people)
        gt_role = [None] * len(people)
        g = gt_by_frame.get(i + 1)
        if g is not None:
            p_xywh = np.column_stack([people.xyxy[:, :2], people.xyxy[:, 2:] - people.xyxy[:, :2]])
            iou = iou_matrix(p_xywh, g[:, 1:])
            for r, c in zip(*linear_sum_assignment(-iou)):
                if iou[r, c] >= 0.5:
                    gt_role[r], gt_side[r] = roles.get(int(g[c, 0]), (None, None))
        is_player = people.class_id == PLAYER_ID
        crop_idx = np.full(len(people), -1)
        crop_idx[is_player] = np.arange(len(crops), len(crops) + is_player.sum())
        new_crops = torso_crops(frame, people.xyxy[is_player])
        if i % fit_stride == 0:
            fit_idx += list(range(len(crops), len(crops) + len(new_crops)))
        crops += new_crops
        frames_out.append({"people": people, "crop_idx": crop_idx, "gt_role": gt_role, "gt_side": gt_side})
    return crops, np.array(fit_idx), frames_out


def best_perm_acc(pairs: list[tuple[int, str]]) -> tuple[float, int]:
    """pairs: (nhãn dự đoán, "left"/"right"). Trả về accuracy với hoán vị tốt nhất và số mẫu."""
    if not pairs:
        return float("nan"), 0
    a = sum((p == 0) == (s == "left") for p, s in pairs)
    return max(a, len(pairs) - a) / len(pairs), len(pairs)


def evaluate(frames_out, preds: np.ndarray, votes: int, sliding: bool):
    """Chạy TeamAssigner trên các frame mẫu. Trả về các cặp (dự đoán, GT) cho: crop, vote, thủ môn."""
    crop_pairs, vote_pairs, gk_pairs = [], [], []
    assigner = TeamAssigner(classifier=None, votes=votes, sliding=sliding)
    for fr in frames_out:
        people, crop_idx = fr["people"], fr["crop_idx"]
        idx = assigner.players_to_classify(people)
        teams = assigner.assign_with_predictions(people, idx, preds[crop_idx[idx]])
        for j in np.flatnonzero(crop_idx >= 0):
            if fr["gt_role"][j] == "player":
                crop_pairs.append((int(preds[crop_idx[j]]), fr["gt_side"][j]))
                vote_pairs.append((int(teams[j]), fr["gt_side"][j]))
        for j in np.flatnonzero(people.class_id == GOALKEEPER_ID):
            if fr["gt_role"][j] == "goalkeeper" and teams[j] in (0, 1):
                gk_pairs.append((int(teams[j]), fr["gt_side"][j]))
    return crop_pairs, vote_pairs, gk_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tracker", default=str(ROOT / "configs" / "botsort_tuned.yaml"))
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=5, help="đánh giá mỗi N frame")
    ap.add_argument("--fit-stride", type=int, default=25, help="crop để fit KMeans lấy mỗi N frame")
    ap.add_argument("--out", default="outputs/team_eval")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    seqs = args.seqs or sorted(p.name for p in root.iterdir() if (p / "img1").is_dir() and (p / "gameinfo.ini").exists())
    seqs = seqs[: args.limit] if args.limit else seqs

    detector = Detector(args.weights, device=args.device)
    # dùng chung cache detection với eval_soccernet.py (cùng imgsz / conf mặc định)
    cache_dir = Path("outputs/det_cache") / f"{Path(args.weights).stem}_{detector.imgsz}_{detector.conf}_{detector.ball_conf}"
    embedders = {"color": ColorEmbedder(), "siglip": SiglipEmbedder(device=args.device)}
    configs = [(e, r, k) for e, r in METHODS for k in CLUSTERS]
    names = [f"{e}{'+' + r if r else ''}, k={k}" for e, r, k in configs]
    # per_seq[name][metric] = list (accuracy, số mẫu) theo sequence
    per_seq = {n: defaultdict(list) for n in names}

    for seq in tqdm(seqs, desc="Sequences"):
        crops, fit_idx, frames_out = collect(root / seq, detector, cache_dir, args.tracker, args.stride, args.fit_stride, args.device)
        feats = {k: emb(crops) for k, emb in embedders.items()}
        for (emb, red, k), name in zip(configs, names):
            clf = TeamClassifier(embedder=embedders[emb], reducer=red, n_clusters=k)
            clf.fit_features(feats[emb][fit_idx])
            preds = clf.predict_features(feats[emb])
            for voting, (votes, sliding) in VOTING.items():
                crop_p, vote_p, gk_p = evaluate(frames_out, preds, votes, sliding)
                per_seq[name][voting].append(best_perm_acc(vote_p))
                if voting == DEFAULT_VOTING:
                    per_seq[name]["crop"].append(best_perm_acc(crop_p))
                    per_seq[name]["gk"].append(best_perm_acc(gk_p))

    def mean(rows):  # trung bình có trọng số theo số mẫu
        n = sum(c for _, c in rows if c)
        return sum(a * c for a, c in rows if c) / n if n else float("nan"), n

    print(f"\n{len(seqs)} sequence, đánh giá mỗi {args.stride} frame. Acc cầu thủ (bỏ phiếu theo track):\n")
    print("| Cách | Từng crop | " + " | ".join(f"Vote {v}" for v in VOTING) + f" | Seq tệ nhất ({DEFAULT_VOTING}) | Thủ môn |")
    print("|---|---|" + "---|" * len(VOTING) + "---|---|")
    summary = {}
    for name in names:
        r = {k: mean(per_seq[name][k]) for k in ["crop", *VOTING, "gk"]}
        worst = min(a for a, c in per_seq[name][DEFAULT_VOTING] if c)
        summary[name] = {k: round(v[0], 3) for k, v in r.items()} | {"worst_seq": round(worst, 3), "gk_n": r["gk"][1]}
        cols = " | ".join(f"{r[v][0]:.3f}" for v in VOTING)
        print(f"| {name} | {r['crop'][0]:.3f} | {cols} | {worst:.3f} | {r['gk'][0]:.3f} (n={r['gk'][1]}) |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_seq_json = {n: {k: [round(a, 3) for a, _ in v] for k, v in d.items()} for n, d in per_seq.items()}
    (out / "summary.json").write_text(json.dumps({"seqs": seqs, "summary": summary, "per_seq": per_seq_json}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
