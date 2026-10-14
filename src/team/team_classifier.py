from collections import Counter, defaultdict, deque

import cv2
import numpy as np
import supervision as sv
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.utils.label_mapping import GOALKEEPER_ID, PLAYER_ID, REFEREE_ID

REFEREE_TEAM = 2
UNKNOWN_TEAM = -1


def torso_crops(frame: np.ndarray, xyxy: np.ndarray) -> list[np.ndarray]:
    """Cắt nửa trên thân cầu thủ (áo), bỏ phần chân/cỏ để embedding tập trung vào màu áo."""
    crops = []
    for x1, y1, x2, y2 in xyxy:
        crops.append(sv.crop_image(frame, [x1, y1, x2, y1 + (y2 - y1) * 0.6]))
    return crops


class SiglipEmbedder:
    def __init__(self, model_id: str = "google/siglip-base-patch16-224", device=None, batch_size: int = 32):
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiglipVisionModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        feats = []
        for i in range(0, len(crops), self.batch_size):
            batch = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops[i : i + self.batch_size]]  # BGR -> RGB
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats.append(self.model(**inputs).pooler_output.cpu().numpy())
        return np.concatenate(feats)


class ColorEmbedder:
    """Baseline "KMeans màu áo": màu Lab trung bình của phần giữa áo, bỏ pixel cỏ."""

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        feats = []
        for c in crops:
            h, w = c.shape[:2]
            c = c[int(h * 0.15) : max(int(h * 0.85), 1), int(w * 0.25) : max(int(w * 0.75), 1)]
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
            grass = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] > 60)
            feats.append(lab[~grass].mean(axis=0) if (~grass).any() else lab.mean(axis=0))
        return np.array(feats)


class TeamClassifier:
    """Embedding -> giảm chiều -> KMeans. Fit một lần cho mỗi video.

    embedder:   "siglip" (mặc định), "color" (baseline màu áo), hoặc một embedder đã khởi tạo để dùng chung
    reducer:    "umap" (mặc định), "pca" hoặc None
    n_clusters: 3 (mặc định) -> 2 cụm lớn nhất là 2 đội, cụm thứ 3 hút trọng tài / thủ môn bị detector nhận nhầm
                thành player (nếu không, KMeans(2) có thể chia "cầu thủ vs trọng tài" thay vì chia 2 đội).
                Khi predict, mỗi crop được gán vào đội có tâm gần nhất.
    """

    def __init__(self, embedder="siglip", reducer: str | None = "umap", n_clusters: int = 3, device=None):
        if isinstance(embedder, str):
            embedder = SiglipEmbedder(device=device) if embedder == "siglip" else ColorEmbedder()
        self.embed = embedder
        if reducer == "umap":
            from umap import UMAP

            self.reducer = UMAP(n_components=3, random_state=0)
        else:
            self.reducer = PCA(n_components=3, random_state=0) if reducer == "pca" else None
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        self.team_centers = None

    def _reduce(self, feats: np.ndarray, fit: bool = False) -> np.ndarray:
        if self.reducer is None:
            return feats
        return self.reducer.fit_transform(feats) if fit else self.reducer.transform(feats)

    def fit(self, crops: list[np.ndarray]):
        self.fit_features(self.embed(crops))

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty(0, dtype=int)
        return self.predict_features(self.embed(crops))

    def fit_features(self, feats: np.ndarray):
        labels = self.kmeans.fit_predict(self._reduce(feats, fit=True))
        biggest = np.argsort(np.bincount(labels, minlength=self.kmeans.n_clusters))[::-1][:2]
        self.team_centers = self.kmeans.cluster_centers_[np.sort(biggest)]

    def predict_features(self, feats: np.ndarray) -> np.ndarray:
        x = self._reduce(feats)
        return np.linalg.norm(x[:, None] - self.team_centers[None], axis=2).argmin(axis=1)


class TeamAssigner:
    """Gán đội theo tracker_id bằng bỏ phiếu qua nhiều frame (ổn định hơn phân loại từng frame).

    - player: đa số trong `votes` dự đoán. sliding=True (mặc định) lấy các dự đoán GẦN NHẤT nên tự sửa khi
      tracker tráo ID giữa 2 cầu thủ khác đội; sliding=False chỉ lấy `votes` dự đoán đầu (rẻ hơn, nhưng
      nhãn bị "đóng băng" sai sau khi tráo ID)
    - goalkeeper: gán vào đội có tâm vị trí gần nhất
    - referee: đội riêng (REFEREE_TEAM)
    """

    def __init__(self, classifier: TeamClassifier | None, votes: int = 15, sliding: bool = True):
        self.classifier = classifier
        self.votes = votes
        self.sliding = sliding
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=votes))

    def players_to_classify(self, people: sv.Detections) -> np.ndarray:
        is_player = people.class_id == PLAYER_ID
        if self.sliding:
            return np.flatnonzero(is_player)
        return np.array([i for i in np.flatnonzero(is_player) if len(self.history[people.tracker_id[i]]) < self.votes], dtype=int)

    def assign(self, frame: np.ndarray, people: sv.Detections) -> np.ndarray:
        if len(people) == 0:
            return np.empty(0, dtype=int)
        idx = self.players_to_classify(people)
        preds = self.classifier.predict(torso_crops(frame, people.xyxy[idx])) if len(idx) else []
        return self.assign_with_predictions(people, idx, preds)

    def assign_with_predictions(self, people: sv.Detections, idx: np.ndarray, preds) -> np.ndarray:
        """Cập nhật phiếu bằng dự đoán của các người ở vị trí idx rồi trả về đội của mọi người trong frame."""
        for i, team in zip(idx, preds):
            self.history[people.tracker_id[i]].append(int(team))

        teams = np.full(len(people), UNKNOWN_TEAM, dtype=int)
        for i in np.flatnonzero(people.class_id == PLAYER_ID):
            hist = self.history[people.tracker_id[i]]
            if hist:
                teams[i] = Counter(hist).most_common(1)[0][0]

        teams[people.class_id == REFEREE_ID] = REFEREE_TEAM
        self._assign_goalkeepers(people, teams)
        return teams

    @staticmethod
    def _assign_goalkeepers(people: sv.Detections, teams: np.ndarray):
        feet = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        centroids = [feet[teams == t].mean(axis=0) if np.any(teams == t) else None for t in (0, 1)]
        if any(c is None for c in centroids):
            return
        for i in np.flatnonzero(people.class_id == GOALKEEPER_ID):
            teams[i] = int(np.argmin([np.linalg.norm(feet[i] - c) for c in centroids]))
