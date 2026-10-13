"""Pipeline đầy đủ trên 1 video: detect -> (SAHI bóng) -> track -> (phân đội) -> vẽ -> ghi video.

    python scripts/infer_video.py --source data/clip.mp4 --weights outputs/best.pt
    python scripts/infer_video.py --source data/clip.mp4 --weights outputs/best.pt \
        --tracker configs/botsort_reid.yaml --sahi --team --mot-out outputs/mot/clip.txt
"""
import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import supervision as sv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detection.ball_sahi import SlicedBallDetector  # noqa: E402
from src.detection.detector import Detector, split_people_ball  # noqa: E402
from src.eval.export_mot import MOTWriter  # noqa: E402
from src.tracking.ball_tracker import BallTracker  # noqa: E402
from src.tracking.trackers import PeopleTracker  # noqa: E402
from src.utils.label_mapping import PLAYER_ID  # noqa: E402
from src.viz.annotate import FootballAnnotator  # noqa: E402


def collect_player_crops(source: str, detector: Detector, stride: int, max_frames: int | None) -> list:
    from src.team.team_classifier import torso_crops

    crops = []
    frames = sv.get_video_frames_generator(source, stride=stride, end=max_frames)
    for frame in tqdm(frames, desc="Lấy crop cầu thủ"):
        dets = detector(frame)
        crops += torso_crops(frame, dets.xyxy[dets.class_id == PLAYER_ID])
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", default="outputs/annotated.mp4")
    ap.add_argument("--tracker", default=str(ROOT / "configs" / "bytetrack.yaml"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ball-conf", type=float, default=0.1)
    ap.add_argument("--sahi", action="store_true", help="sliced inference cho bóng")
    ap.add_argument("--team", action="store_true", help="phân đội bằng SigLIP")
    ap.add_argument("--team-stride", type=int, default=30, help="lấy crop mỗi N frame để fit phân đội")
    ap.add_argument("--mot-out", default=None, help="ghi kết quả tracking cầu thủ dạng MOT")
    ap.add_argument("--max-frames", type=int, default=None, help="chỉ chạy N frame đầu (debug)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    detector = Detector(args.weights, imgsz=args.imgsz, conf=args.conf, ball_conf=args.ball_conf, device=args.device)
    ball_detector = SlicedBallDetector(detector) if args.sahi else None
    tracker = PeopleTracker(args.tracker, device=args.device)
    ball_tracker = BallTracker()
    annotator = FootballAnnotator()

    team_assigner = None
    if args.team:
        from src.team.team_classifier import TeamAssigner, TeamClassifier

        classifier = TeamClassifier(device=args.device)
        classifier.fit(collect_player_crops(args.source, detector, args.team_stride, args.max_frames))
        team_assigner = TeamAssigner(classifier)

    video_info = sv.VideoInfo.from_video_path(args.source)
    if args.max_frames:
        video_info.total_frames = min(video_info.total_frames or args.max_frames, args.max_frames)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    frames = sv.get_video_frames_generator(args.source, end=args.max_frames)
    with sv.VideoSink(args.output, video_info) as sink, (MOTWriter(args.mot_out) if args.mot_out else nullcontext()) as mot:
        for i, frame in enumerate(tqdm(frames, total=video_info.total_frames, desc="Tracking")):
            people, ball = split_people_ball(detector(frame))
            if ball_detector:
                ball = ball_detector(frame)

            people = tracker.update(people, frame)
            ball = ball_tracker.update(ball)
            teams = team_assigner.assign(frame, people) if team_assigner else None

            sink.write_frame(annotator.annotate(frame, people, ball, teams))
            if mot:
                mot.write(i + 1, people)

    print(f"Đã ghi {args.output}")


if __name__ == "__main__":
    main()
