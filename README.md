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
    --sahi --team --mot-out outputs/mot/clip.txt

# Đánh giá trên subset SoccerNet-Tracking
python scripts/eval_soccernet.py --data-root data/SoccerNet/tracking/test --weights outputs/best.pt \
    --trackers configs/bytetrack.yaml configs/botsort_tuned.yaml --limit 10
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

### Tracking

Đo bằng `scripts/eval_soccernet.py` (TrackEval, chỉ tính người, không tính bóng). Mọi tracker nhận **cùng một bộ detection** (đã cache) nên khác biệt chỉ đến từ tracker. Tham số được chọn trên SNMOT-061 → 065 (tập tinh chỉnh), còn bảng dưới đo trên **SNMOT-066 → 070, tập không dùng khi tinh chỉnh**. FPS là FPS của riêng tracker (không tính detect) trên GTX 1050 Ti.

| Tracker | HOTA | IDF1 | IDSW | MOTA | FPS tracker |
|---|---|---|---|---|---|
| ByteTrack (mặc định Ultralytics) | 43.1 | 46.1 | 1113 | 71.1 | 356 |
| BoT-SORT (GMC) | 46.4 | 52.3 | 921 | 70.9 | 29 |
| BoT-SORT + ReID `yolo26n-cls` | 47.6 | 53.7 | 1251 | 67.7 | 8 |
| ByteTrack, `new_track_thresh` 0.8 | 51.1 | 59.3 | 318 | 78.9 | **547** |
| **BoT-SORT, `new_track_thresh` 0.8** (`configs/botsort_tuned.yaml`, mặc định) | **53.6** | 64.5 | 257 | **79.1** | 31 |
| BoT-SORT, `new_track_thresh` 0.85 | 52.8 | **64.8** | 229 | 76.8 | 31 |
| BoT-SORT, `new_track_thresh` 0.9 | 48.5 | 63.2 | **172** | 67.1 | 31 |
| BoT-SORT + ReID, `new_track_thresh` 0.8 | 51.9 | 61.8 | 459 | 77.3 | 8 |

Kết quả trên tập tinh chỉnh cùng xu hướng (BoT-SORT mặc định 47.2 → 52.6 HOTA với ngưỡng 0.8), nên cấu hình được chọn không bị overfit.

**Thay đổi quan trọng nhất là một tham số:** `new_track_thresh` 0.25 → 0.8 giảm ID switch 72% (921 → 257) và tăng HOTA 7.2 điểm. GMC đóng góp thêm khoảng 2.5 HOTA, còn ReID làm kết quả xấu đi.

#### Vì sao: phân loại ID switch (`scripts/analyze_idsw.py`)

Mỗi frame ghép GT ↔ track (Hungarian, IoU ≥ 0.5, ưu tiên giữ cặp cũ như TrackEval) rồi phân loại mỗi lần một người đổi ID. Tổng số đếm được khớp với TrackEval (899 so với 921).

| Nguyên nhân | BoT-SORT mặc định | BoT-SORT 0.8 |
|---|---|---|
| Track cũ đứt, track mới thay ngay (frame trước còn đúng) | **443** | **19** |
| Track mới sau khi mất dấu | 209 | 109 |
| Tráo ID với người khác | 132 | 90 |
| 2 track cùng bám 1 người | 115 | 34 |
| *Trong đó: xảy ra khi đang chồng lấn người khác* | *401* | *150* |

- **Nguyên nhân chính là track thừa từ detection conf thấp.** Khi một detection không ghép được với track (do dự đoán Kalman lệch), ngưỡng 0.25 cho phép nó sinh ngay một ID mới, và ID mới này thường thắng track cũ ở frame sau. Detector có precision 0.99 với người nên đòi conf ≥ 0.8 mới tạo track chỉ loại nhiễu mà hầu như không bỏ sót người thật. Tới 0.9 thì bắt đầu bỏ sót cầu thủ mới vào khung hình (MOTA giảm 12 điểm).
- **Giả thuyết bị loại:** tắt `fuse_score` làm HOTA giảm 47.2 → 44.0; hạ conf người xuống 0.1 để dùng tầng ghép thứ 2 của ByteTrack thì HOTA không tăng mà tracker chậm hơn.
- **Lỗi còn lại cần ReID thật:** sau khi chỉnh ngưỡng, lỗi chủ yếu là mất dấu lâu rồi ra ID mới (trung vị 67 frame ≈ 2.7 s, dài hơn `track_buffer` 50 frame: cầu thủ ra khỏi khung hình rồi quay lại) và tráo ID khi chồng lấn. IoU + Kalman không giải quyết được hai loại này.

**Hạn chế:** ReID trong bảng dùng `yolo26n-cls.pt`, một model phân loại ImageNet, không phải model ReID được train cho người. Embedding của nó không phân biệt được cầu thủ cùng đội mặc áo giống nhau, nên ReID tăng tráo ID thay vì giảm. Kết luận "ReID không giúp" chỉ đúng với model này. Cần thử lại với model ReID thật (vd OSNet qua BoxMOT).

## Phân tích lỗi & hạn chế

<!-- TODO: ID switch xảy ra khi nào, bóng mất khi nào, phân đội sai khi nào -->
