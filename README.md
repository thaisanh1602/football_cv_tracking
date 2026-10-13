# football-cv-tracking

Detection + tracking cầu thủ / thủ môn / trọng tài / bóng trên video bóng đá broadcast, đánh giá trên SoccerNet-Tracking.

<!-- TODO: GIF demo ở đây -->

## Pipeline

```
frame ─► YOLO (imgsz 1280) ─┬─ người ─► ByteTrack / BoT-SORT (+ReID) ─► phân đội (SigLIP → UMAP → KMeans, vote theo track ID)
                            └─ bóng  ─► SAHI (tile 640) ─► BallTracker (1 bóng/frame, lọc nhảy) ─┴─► vẽ + xuất MOT
```

## Cấu trúc

```
configs/        data.yaml, bytetrack.yaml, botsort.yaml, botsort_reid.yaml
src/
  detection/    detector.py (YOLO -> sv.Detections, map class), ball_sahi.py (sliced inference cho bóng)
  tracking/     trackers.py (bọc tracker Ultralytics), ball_tracker.py
  team/         team_classifier.py (SigLIP + UMAP + KMeans, TeamAssigner)
  eval/         export_mot.py, run_trackeval.py (HOTA / IDF1 / IDSW)
  viz/          annotate.py
  utils/        label_mapping.py
scripts/        prepare_dataset.py, train.py, infer_video.py, eval_soccernet.py
notebooks/      kaggle_train.ipynb, kaggle_eval.ipynb
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

Train trên Kaggle: `notebooks/kaggle_train.ipynb`. Code / debug ở local, chạy được với GPU 4GB.

```bash
# Chạy thử pipeline bằng model COCO trước khi có model 4 class (person -> player, sports ball -> ball)
python scripts/infer_video.py --source data/clip.mp4 --weights yolo11n.pt --max-frames 100

# Đầy đủ
python scripts/infer_video.py --source data/clip.mp4 --weights outputs/best.pt \
    --tracker configs/botsort_reid.yaml --sahi --team --mot-out outputs/mot/clip.txt

# Đánh giá trên subset SoccerNet-Tracking
python scripts/eval_soccernet.py --data-root data/SoccerNet/tracking/test --weights outputs/best.pt \
    --tracker configs/bytetrack.yaml --limit 10
```

## Kết quả

### Detection (val)

| Class | AP50 | AP50-95 |
|---|---|---|
| player | | |
| goalkeeper | | |
| referee | | |
| ball | | |

### Bóng: full-frame vs SAHI

| | Ball recall | Ball AP50 | FPS |
|---|---|---|---|
| Full-frame 1280 | | | |
| SAHI tile 640 | | | |

### Tracking (SoccerNet-Tracking test, 10 sequence)

| Tracker | HOTA | IDF1 | IDSW | MOTA | FPS |
|---|---|---|---|---|---|
| ByteTrack | | | | | |
| BoT-SORT (GMC, no ReID) | | | | | |
| BoT-SORT + ReID | | | | | |

Mốc tham khảo: `yolo11n.pt` COCO + ByteTrack trên SNMOT-061 (train) đạt HOTA 40.8 / IDF1 43.3 / IDSW 190.

## Phân tích lỗi & hạn chế

<!-- TODO: ID switch xảy ra khi nào, bóng mất khi nào, phân đội sai khi nào -->
