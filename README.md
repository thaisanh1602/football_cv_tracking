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

38 ảnh val, `imgsz 1280`.

| Class | Instances | P | R | AP50 | AP50-95 |
|---|---|---|---|---|---|
| **all** | 905 | 0.981 | 0.855 | **0.882** | **0.654** |
| player | 754 | 0.990 | 0.984 | 0.994 | 0.844 |
| goalkeeper | 27 | 0.984 | 0.926 | 0.987 | 0.777 |
| referee | 89 | 0.960 | 0.966 | 0.976 | 0.726 |
| ball | 35 | 0.988 | 0.543 | 0.569 | 0.269 |

Người được detect gần như hoàn hảo. Điểm yếu là bóng: precision 0.988 nhưng recall chỉ 0.543. Nghĩa là box bóng nào model đưa ra thì hầu như đúng, nhưng gần một nửa số bóng bị bỏ sót, vì bóng chỉ chiếm vài pixel ở khung hình broadcast. Đây là lý do dùng SAHI cho bóng (xem bảng bên dưới).

Đường cong train (loss / mAP theo epoch) và confusion matrix chuẩn hóa trên val:

![results](assets/results.png)

![confusion matrix](assets/confusion_matrix_normalized.png)

### Bóng: full-frame vs SAHI

Đo bằng `scripts/eval_ball.py` trên SoccerNet-Tracking SNMOT-061 → 065, mỗi 5 frame lấy 1 (750 frame). Detector train trên Roboflow nên không bị lộ dữ liệu. Bóng GT có kích thước trung vị 14×12 px nên một detection được tính là đúng khi tâm cách tâm GT ≤ 15 px. FPS đo phần detection trên GTX 1050 Ti.

| Cấu hình | Recall | Precision | F1 | FPS |
|---|---|---|---|---|
| Full-frame, conf bóng 0.25 | 0.485 | 0.897 | 0.630 | 5.7 |
| SAHI tile 640 → imgsz 640 | 0.420 | 0.754 | 0.539 | 1.5 |
| SAHI tile 960 → imgsz 640 | 0.549 | 0.821 | 0.658 | 1.5 |
| Full-frame, conf bóng 0.1 | 0.559 | 0.789 | 0.654 | 4.4 |
| **Full-frame, conf bóng 0.1 + BallTracker** | 0.552 | **0.894** | **0.682** | **4.4** |
| SAHI 960 + BallTracker | 0.546 | 0.887 | 0.676 | 1.5 |
| Full + SAHI 960 + BallTracker | 0.550 | 0.866 | 0.673 | 1.5 |

Kết luận:
- **Scale của tile quyết định SAHI có giúp hay không.** Model train ở 1920→1280 (×0.67). Tile 640 → 640 làm bóng to gấp 1.5 lần so với lúc train, và recall giảm thay vì tăng. Tile 960 → 640 giữ đúng ×0.67 thì recall tăng 0.42 → 0.549.
- **Hạ conf + lọc bằng tracker hiệu quả ngang SAHI mà nhanh gấp khoảng 3 lần.** Hạ ngưỡng bóng xuống 0.1 kéo recall lên mức của SAHI. BallTracker (giữ 1 bóng/frame, loại điểm nhảy xa) đưa precision từ 0.789 lên 0.894. Đây là cấu hình mặc định; SAHI để tùy chọn `--sahi`.
- **Recall chững lại ở khoảng 0.55 với mọi cấu hình**, nên giới hạn không nằm ở độ phân giải. Giả thuyết: các frame bị bỏ sót là frame khó (bóng bị che, bị nhòe khi bay nhanh, nằm sát vạch trắng) và tập train có quá ít bóng (val chỉ có 35 instance). Nếu đúng, thêm dữ liệu bóng để train sẽ có tác dụng hơn đổi cách inference. Cần xem lại các frame bị bỏ sót để xác nhận.

### Tracking (SoccerNet-Tracking test, 10 sequence)

| Tracker | HOTA | IDF1 | IDSW | MOTA | FPS |
|---|---|---|---|---|---|
| ByteTrack | | | | | |
| BoT-SORT (GMC, no ReID) | | | | | |
| BoT-SORT + ReID | | | | | |

Mốc tham khảo: `yolo11n.pt` COCO + ByteTrack trên SNMOT-061 (train) đạt HOTA 40.8 / IDF1 43.3 / IDSW 190.

## Phân tích lỗi & hạn chế

<!-- TODO: ID switch xảy ra khi nào, bóng mất khi nào, phân đội sai khi nào -->
