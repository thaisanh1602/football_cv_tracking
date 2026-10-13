from pathlib import Path

import numpy as np

# TrackEval còn dùng np.float / np.int / np.bool đã bị xoá từ numpy 1.24
for _alias, _type in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _type)


def run_trackeval(gt_root: str | Path, trackers_root: str | Path, tracker_name: str, seqs: list[str]) -> dict:
    """Tính HOTA / IDF1 / IDSW / MOTA bằng TrackEval.

    gt_root:        thư mục chứa các sequence SoccerNet, mỗi seq có gt/gt.txt và seqinfo.ini
    trackers_root:  kết quả nằm ở <trackers_root>/<tracker_name>/data/<seq>.txt
    """
    try:
        import trackeval
    except ImportError as e:
        raise SystemExit("Cần cài TrackEval: pip install git+https://github.com/JonathonLuiten/TrackEval.git") from e

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(USE_PARALLEL=False, PRINT_CONFIG=False, PLOT_CURVES=False, OUTPUT_DETAILED=False)

    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(
        GT_FOLDER=str(gt_root),
        TRACKERS_FOLDER=str(trackers_root),
        TRACKERS_TO_EVAL=[tracker_name],
        BENCHMARK="SNMOT",
        SKIP_SPLIT_FOL=True,
        DO_PREPROC=False,  # giống sn-tracking: GT SoccerNet không dùng class pedestrian/distractor của MOT17
        SEQ_INFO={s: None for s in seqs},  # None -> đọc seqLength từ seqinfo.ini
        PRINT_CONFIG=False,
    )

    metric_cfg = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(metric_cfg), trackeval.metrics.CLEAR(metric_cfg), trackeval.metrics.Identity(metric_cfg)]
    res, _ = trackeval.Evaluator(eval_cfg).evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)

    r = res["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"]["pedestrian"]
    return {
        "HOTA": float(np.mean(r["HOTA"]["HOTA"])) * 100,
        "IDF1": float(r["Identity"]["IDF1"]) * 100,
        "IDSW": int(r["CLEAR"]["IDSW"]),
        "MOTA": float(r["CLEAR"]["MOTA"]) * 100,
    }
