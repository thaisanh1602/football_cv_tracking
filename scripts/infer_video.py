"""Pipeline đầy đủ trên 1 video: detect -> (SAHI bóng) -> track -> (phân đội) -> vẽ -> ghi video.

    python scripts/infer_video.py --source data/clip.mp4 --weights outputs/best.pt
    python scripts/infer_video.py --source data/clip.mp4 --weights outputs/best.pt \
        --sahi --team --mot-out outputs/mot/clip.txt
"""
import argparse
import sys
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import supervision as sv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detection.ball_sahi import SlicedBallDetector  # noqa: E402
from src.detection.detector import Detector, split_people_ball  # noqa: E402
from src.eval.export_mot import MOTWriter  # noqa: E402
from src.tracking.ball_tracker import BallTracker  # noqa: E402
from src.tracking.trackers import PeopleTracker  # noqa: E402
from src.utils.label_mapping import PLAYER_ID, REFEREE_ID  # noqa: E402
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
    ap.add_argument("--tracker", default=str(ROOT / "configs" / "botsort_tuned.yaml"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ball-conf", type=float, default=0.1)
    ap.add_argument("--sahi", action="store_true", help="sliced inference cho bóng")
    ap.add_argument("--team", action="store_true", help="phân đội bằng SigLIP")
    ap.add_argument("--team-stride", type=int, default=30, help="lấy crop mỗi N frame để fit phân đội")
    ap.add_argument("--mot-out", default=None, help="ghi kết quả tracking cầu thủ dạng MOT")
    ap.add_argument("--pitch-weights", default=None, help="model keypoint sân -> vẽ radar 2D")
    ap.add_argument("--pitch-imgsz", type=int, default=1280, help="imgsz lúc train model keypoint sân")
    ap.add_argument("--no-smooth", action="store_true", help="tắt làm mượt homography (để so sánh độ rung)")
    ap.add_argument("--no-radar-filter", action="store_true", help="tắt Kalman theo từng cầu thủ trên radar (bản 'trước' để so sánh)")
    ap.add_argument("--radar-layout", choices=["stack", "overlay"], default="stack", help="radar dưới video hoặc chồng lên")
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

    pitch_mapper = radar = None
    if args.pitch_weights:
        from src.pitch.homography import PitchMapper
        from src.pitch.keypoints import PitchKeypointDetector
        from src.pitch.radar import Radar, overlay_radar, stack_radar

        from src.pitch.kinematics import OnlineKinematics

        from src.stabilize.overlay_filter import KalmanCV, TrackFilterBank

        kinematics = OnlineKinematics(fps=video_info.fps)
        # Kalman vận tốc không đổi theo từng cầu thủ: trên GSR giảm rung radar 10.1 -> 1.0 cm/frame², sai số +3 cm
        radar_filter = None if args.no_radar_filter else TrackFilterBank(lambda: KalmanCV(fps=video_info.fps, accel_std=2.0), fps=video_info.fps)
        trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=int(video_info.fps)))
        pitch_mapper = PitchMapper(PitchKeypointDetector(args.pitch_weights, imgsz=args.pitch_imgsz, device=args.device), smooth=not args.no_smooth, fps=video_info.fps)
        radar = Radar()
        if args.radar_layout == "stack":  # khung hình cao thêm phần radar
            probe = stack_radar(np.zeros((video_info.height, video_info.width, 3), np.uint8), radar.base)
            video_info.height = probe.shape[0]

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

            speeds = None
            radar_img = None
            if pitch_mapper:
                if pitch_mapper.update(frame) is not None:
                    feet = pitch_mapper.to_pitch(people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)) if len(people) else np.empty((0, 2))
                    ball_m = pitch_mapper.to_pitch(ball.get_anchors_coordinates(sv.Position.CENTER)) if len(ball) else None
                    ids = people.tracker_id if len(people) else []
                    speeds = kinematics.update(i, ids, feet)  # tốc độ đã tự làm mượt (cửa sổ 2 s) trên vị trí chưa lọc
                    if radar_filter is not None and len(feet):
                        feet = radar_filter.update(i, ids, feet)
                    for tid, p in zip(ids, feet):
                        trails[tid].append(p)
                    radar_teams = teams if teams is not None else np.where(people.class_id == REFEREE_ID, 2, 3)
                    radar_img = radar.draw(feet, radar_teams, ball_m, [np.array(trails[t]) for t in ids])
                else:
                    radar_img = radar.empty()

            out = annotator.annotate(frame, people, ball, teams, speeds)
            if radar_img is not None:
                out = stack_radar(out, radar_img) if args.radar_layout == "stack" else overlay_radar(out, radar_img)
            sink.write_frame(out)
            if mot:
                mot.write(i + 1, people)

    print(f"Đã ghi {args.output}")


if __name__ == "__main__":
    main()
