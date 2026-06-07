from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# Pesos por grupo de feature — calibrados para domínio de moda
_FEATURE_WEIGHTS = {
    "brand":    2.0,
    "category": 1.8,
    "name":     1.5,
    "size":     1.2,
    "color":    0.8,
    "price":    0.5,
    "rating":   0.7,
}


class ContentBasedFilter:
    """
    Representação vetorial ponderada de produtos.
    one-hot encoding com pesos por importância de feature.
    """

    def __init__(self) -> None:
        self._feature_matrix: Optional[np.ndarray] = None
        self._product_index: dict[int, int]         = {}   # product_id → linha
        self._index_to_pid: dict[int, int]          = {}   # linha → product_id
        self._products_df: Optional[pd.DataFrame]   = None

    @property
    def is_trained(self) -> bool:
        return self._feature_matrix is not None

    @staticmethod
    def _onehot(series: pd.Series, weight: float) -> np.ndarray:
        """One-hot encoding de uma série categórica, escalonado pelo peso."""
        dummies = pd.get_dummies(series, prefix=series.name)
        return dummies.values.astype(float) * weight

    def train(self, products_df: pd.DataFrame) -> None:
       
        logger.info("Construindo índice de Filtragem Baseada em Conteúdo (feature vectors ponderados)...")

        df = products_df.copy().reset_index(drop=True)

        parts: list[np.ndarray] = [
            self._onehot(df["brand"],        _FEATURE_WEIGHTS["brand"]),
            self._onehot(df["category"],     _FEATURE_WEIGHTS["category"]),
            self._onehot(df["product_name"], _FEATURE_WEIGHTS["name"]),
            self._onehot(df["size"],         _FEATURE_WEIGHTS["size"]),
            self._onehot(df["color"],        _FEATURE_WEIGHTS["color"]),
        ]

        # Features contínuas normalizadas
        scaler = MinMaxScaler()
        cont = scaler.fit_transform(
            df[["price", "avg_rating"]].fillna(df[["price", "avg_rating"]].mean())
        )
        parts.append(cont[:, 0:1] * _FEATURE_WEIGHTS["price"])
        parts.append(cont[:, 1:2] * _FEATURE_WEIGHTS["rating"])

        feature_matrix = np.hstack(parts)

        # Normaliza cada vetor para que a similaridade cosseno seja bem definida
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feature_matrix = feature_matrix / norms

        with _lock:
            self._feature_matrix = feature_matrix
            self._product_index  = {int(row["product_id"]): idx for idx, row in df.iterrows()}
            self._index_to_pid   = {idx: int(row["product_id"]) for idx, row in df.iterrows()}
            self._products_df    = df

        n_features = feature_matrix.shape[1]
        logger.info(f"Índice de conteúdo construído | Produtos: {len(df)} | Features: {n_features}")


    def get_similar_products(self, product_id: int, top_n: int = 20) -> list[tuple[int, float]]:
        """Retorna os top_n produtos mais similares ao product_id dado."""
        if not self.is_trained:
            return []
        idx = self._product_index.get(product_id)
        if idx is None:
            return []

        query  = self._feature_matrix[idx].reshape(1, -1)
        sims   = cosine_similarity(query, self._feature_matrix)[0]
        sims[idx] = -1  # exclui o próprio produto

        top_idx = np.argsort(sims)[::-1][:top_n]
        return [(self._index_to_pid[i], float(sims[i])) for i in top_idx]

    def score_for_user(
        self,
        user_rated_products: list[tuple[int, float]],
        candidate_ids: list[int],
    ) -> dict[int, float]:
        """
        Constrói o perfil vetorial do usuário como média ponderada pelos ratings
        (apenas produtos com rating >= 3 contribuem positivamente; ratings baixos
        subtraem do perfil — estratégia de aversão ativa).
        Retorna similaridade cosseno entre o perfil e cada candidato.
        """
        if not self.is_trained or not user_rated_products:
            return {pid: 0.0 for pid in candidate_ids}

        user_vec = np.zeros(self._feature_matrix.shape[1])
        for pid, rating in user_rated_products:
            idx = self._product_index.get(pid)
            if idx is None:
                continue
            # Transformação: ratings centralizados em 3 (neutro), de -1 a +1
            w = (rating - 3.0) / 2.0
            user_vec += w * self._feature_matrix[idx]

        norm = np.linalg.norm(user_vec)
        if norm < 1e-9:
            return {pid: 0.0 for pid in candidate_ids}
        user_vec = user_vec / norm

        # Mapeamento O(1): pid → posição no candidate array
        pid_to_pos = {
            pid: pos
            for pos, pid in enumerate(candidate_ids)
            if pid in self._product_index
        }
        valid_pids = [pid for pid in candidate_ids if pid in self._product_index]
        if not valid_pids:
            return {pid: 0.0 for pid in candidate_ids}

        candidate_rows = np.array([self._product_index[pid] for pid in valid_pids])
        candidate_mat  = self._feature_matrix[candidate_rows]
        sims           = cosine_similarity(user_vec.reshape(1, -1), candidate_mat)[0]

        scores = {pid: 0.0 for pid in candidate_ids}
        for i, pid in enumerate(valid_pids):
            scores[pid] = float(sims[i])
        return scores


# Instância global
content_filter = ContentBasedFilter()
