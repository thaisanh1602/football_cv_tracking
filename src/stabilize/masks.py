"""Mask người (YOLO-seg) và ổn định mask theo thời gian.

Mask từng frame nhấp nháy: viền nhảy, tay chân lúc có lúc mất. Ổn định theo từng track trong một khung chuẩn
hoá theo box ("canvas"): cắt mask theo box rồi resize về kích thước cố định. Trong canvas, chuyển động camera,
việc cầu thủ chạy và co giãn theo khoảng cách đã bị trừ đi; phần còn lại là thay đổi hình dạng. Mask ổn định
= trung bình trượt mũ của xác suất mask trong canvas, rồi chiếu ngược về box hiện tại.

Không có GT mask nên đo 2 trục:
- flicker = 1 - IoU(mask_t, mask_{t-1}) trong canvas (càng thấp càng ít nhấp nháy)
- fidelity = IoU(mask đã ổn định, mask gốc của model) (ổn định quá mạnh -> mask "đông cứng", tay chân không theo kịp)
"""
import cv2
import numpy as np

CANVAS_WH = (48, 96)  # rộng x cao, xấp xỉ tỷ lệ người đứng
MARGIN = 0.1  # nới box 10% mỗi phía để không cắt mất tay chân


def _crop_box(box, shape):
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    h, w = shape[:2]
    return max(int(x1 - mx), 0), max(int(y1 - my), 0), min(int(x2 + mx) + 1, w), min(int(y2 + my) + 1, h)


def to_canvas(mask: np.ndarray, box) -> np.ndarray:
    """Mask toàn khung (bool/float) -> canvas float32 [0, 1] theo box."""
    x1, y1, x2, y2 = _crop_box(box, mask.shape)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(CANVAS_WH[::-1], np.float32)
    return cv2.resize(mask[y1:y2, x1:x2].astype(np.float32), CANVAS_WH, interpolation=cv2.INTER_AREA)


def from_canvas(canvas: np.ndarray, box, shape) -> np.ndarray:
    """Canvas -> mask bool toàn khung, đặt lại vào box hiện tại."""
    out = np.zeros(shape[:2], bool)
    x1, y1, x2, y2 = _crop_box(box, shape)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = cv2.resize(canvas, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR) > 0.5
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


class PersonSegmenter:
    """YOLO-seg pretrained COCO, chỉ class person (0). Trả về box và mask ở độ phân giải gốc."""

    def __init__(self, weights: str = "yolo11m-seg.pt", imgsz: int = 1280, conf: float = 0.25, device=None):
        from ultralytics import YOLO

        self.model, self.imgsz, self.conf, self.device = YOLO(weights), imgsz, conf, device

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=[0], retina_masks=True, device=self.device, verbose=False)[0]
        if r.masks is None:
            return np.empty((0, 4)), np.empty((0, *frame.shape[:2]), bool)
        return r.boxes.xyxy.cpu().numpy(), r.masks.data.cpu().numpy() > 0.5


def match_masks(seg_boxes: np.ndarray, seg_masks: np.ndarray, boxes: np.ndarray, iou_thr: float = 0.5) -> list:
    """Gán mask cho từng box (của tracker / GT) theo IoU box lớn nhất. Không khớp -> None."""
    from scipy.optimize import linear_sum_assignment

    out = [None] * len(boxes)
    if len(seg_boxes) == 0 or len(boxes) == 0:
        return out
    lt = np.maximum(boxes[:, None, :2], seg_boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], seg_boxes[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
    m = inter / (area(boxes)[:, None] + area(seg_boxes)[None] - inter + 1e-9)
    for r, c in zip(*linear_sum_assignment(-m)):
        if m[r, c] >= iou_thr:
            out[r] = seg_masks[c]
    return out


class MaskStabilizer:
    """EMA xác suất mask trong canvas theo từng track. alpha = trọng số frame mới (1 = không ổn định)."""

    def __init__(self, alpha: float = 0.5, max_gap: int = 12):
        self.alpha, self.max_gap = alpha, max_gap
        self.state: dict[int, tuple[int, np.ndarray]] = {}

    def update_canvas(self, frame_idx: int, tid: int, canvas: np.ndarray) -> np.ndarray:
        prev = self.state.get(tid)
        if prev is None or frame_idx - prev[0] > self.max_gap:
            s = canvas
        else:
            s = self.alpha * canvas + (1 - self.alpha) * prev[1]
        self.state[tid] = (frame_idx, s)
        return s

    def update(self, frame_idx: int, tid: int, mask: np.ndarray, box) -> np.ndarray:
        """Mask toàn khung của 1 track -> mask đã ổn định (toàn khung, bool)."""
        return from_canvas(self.update_canvas(frame_idx, tid, to_canvas(mask, box)), box, mask.shape)
