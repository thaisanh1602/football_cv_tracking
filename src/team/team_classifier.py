from collections import Counter, defaultdict

import numpy as np
import supervision as sv
import torch
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, SiglipVisionModel

from src.utils.label_mapping import GOALKEEPER_ID, PLAYER_ID, REFEREE_ID

try:
    from umap import UMAP
except ImportError:  # chưa cài umap-learn thì tạm dùng PCA
    from sklearn.decomposition import PCA as UMAP

REFEREE_TEAM = 2
UNKNOWN_TEAM = -1


def torso_crops(frame: np.ndarray, xyxy: np.ndarray) -> list[np.ndarray]:
    """Cắt nửa trên thân cầu thủ (áo), bỏ phần chân/cỏ để embedding tập trung vào màu áo."""
    crops = []
    for x1, y1, x2, y2 in xyxy:
        crops.append(sv.crop_image(frame, [x1, y1, x2, y1 + (y2 - y1) * 0.6]))
    return crops


class TeamClassifier:
    """SigLIP embedding -> UMAP -> KMeans(2). Fit một lần cho mỗi video."""

    def __init__(self, model_id: str = "google/siglip-base-patch16-224", device=None, batch_size: int = 32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiglipVisionModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.batch_size = batch_size
        self.reducer = UMAP(n_components=3)
        self.kmeans = KMeans(n_clusters=2, n_init=10)

    @torch.no_grad()
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        feats = []
        for i in range(0, len(crops), self.batch_size):
            batch = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops[i : i + self.batch_size]]  # BGR -> RGB
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats.append(self.model(**inputs).pooler_output.cpu().numpy())
        return np.concatenate(feats)

    def fit(self, crops: list[np.ndarray]):
        self.kmeans.fit(self.reducer.fit_transform(self.embed(crops)))

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty(0, dtype=int)
        return self.kmeans.predict(self.reducer.transform(self.embed(crops)))


class TeamAssigner:
    """Gán đội theo tracker_id bằng bỏ phiếu qua nhiều frame (ổn định hơn phân loại từng frame).

    - player: bỏ phiếu, mỗi track chỉ phân loại tới khi đủ `votes_needed` phiếu để tiết kiệm GPU
    - goalkeeper: gán vào đội có tâm vị trí gần nhất
    - referee: đội riêng (REFEREE_TEAM)
    """

    def __init__(self, classifier: TeamClassifier, votes_needed: int = 10):
        self.classifier = classifier
        self.votes_needed = votes_needed
        self.votes: dict[int, Counter] = defaultdict(Counter)

    def assign(self, frame: np.ndarray, people: sv.Detections) -> np.ndarray:
        teams = np.full(len(people), UNKNOWN_TEAM, dtype=int)
        if len(people) == 0:
            return teams

        is_player = people.class_id == PLAYER_ID
        need = [i for i in np.flatnonzero(is_player) if sum(self.votes[people.tracker_id[i]].values()) < self.votes_needed]
        if need:
            preds = self.classifier.predict(torso_crops(frame, people.xyxy[need]))
            for i, team in zip(need, preds):
                self.votes[people.tracker_id[i]][int(team)] += 1

        for i in np.flatnonzero(is_player):
            votes = self.votes[people.tracker_id[i]]
            if votes:
                teams[i] = votes.most_common(1)[0][0]

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
