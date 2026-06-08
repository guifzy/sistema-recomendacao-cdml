from __future__ import annotations

import logging
import threading
from typing import Optional

import pandas as pd
from surprise import SVD, KNNBaseline, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class CollaborativeFilter:

    def __init__(self) -> None:
        self._svd: Optional[SVD]            = None
        self._knn: Optional[KNNBaseline]    = None
        self._trained_user_ids: set[int]    = set()
        self._rmse_svd: Optional[float]     = None
        self._rmse_knn: Optional[float]     = None
        self._svd_weight: float             = 0.65

    @property
    def is_trained(self) -> bool:
        return self._svd is not None

    @property
    def rmse(self) -> Optional[float]:
        """RMSE médio ponderado dos dois modelos."""
        if self._rmse_svd is None:
            return None
        if self._rmse_knn is None:
            return self._rmse_svd
        return round(
            self._svd_weight * self._rmse_svd + (1 - self._svd_weight) * self._rmse_knn, 4
        )

    def train(self, df: pd.DataFrame) -> None:
        n_items = df["product_id"].nunique()
        # n_factors calibrado: sqrt(n_items), mínimo 20, máximo 80
        n_factors = max(20, min(80, int(n_items ** 0.5)))
        logger.info(f"Treinando SVD (n_factors={n_factors}) para {n_items} itens...")

        reader   = Reader(rating_scale=(1.0, 5.0))
        data     = Dataset.load_from_df(df[["user_id", "product_id", "rating"]], reader)
        trainset, testset = train_test_split(data, test_size=0.15, random_state=42)

        # svd
        svd = SVD(
            n_factors=n_factors,
            n_epochs=30,
            lr_all=0.007,
            reg_all=0.05,   # regularização maior evita overfitting em catálogo pequeno
            biased=True,    # inclui baseline (média global + desvio por usuário/item)
            random_state=42,
            verbose=False,
        )
        svd.fit(trainset)
        svd_preds = svd.test(testset)
        rmse_svd  = accuracy.rmse(svd_preds, verbose=False)
        logger.info(f"SVD treinado | RMSE = {rmse_svd:.4f}")

        # KNNBaseline item-item 
        # Pearson shrunk é mais robusta que cosseno para ratings com poucos dados
        knn_options = {
            "name":       "pearson_baseline",
            "shrinkage":  100,   # penaliza estimativas baseadas em poucas co-avaliações
            "user_based": False, # item-item: foca em quais produtos co-ocorrem
        }
        knn = KNNBaseline(
            k=40,
            min_k=3,
            sim_options=knn_options,
            verbose=False,
        )
        knn.fit(trainset)
        knn_preds = knn.test(testset)
        rmse_knn  = accuracy.rmse(knn_preds, verbose=False)
        logger.info(f"KNNBaseline (item-item) treinado | RMSE = {rmse_knn:.4f}")

        # Peso do SVD inversamente proporcional ao seu RMSE
        total = rmse_svd + rmse_knn
        svd_w = rmse_knn / total if total > 0 else 0.65

        with _lock:
            self._svd              = svd
            self._knn              = knn
            self._svd_weight       = svd_w
            self._rmse_svd         = round(rmse_svd, 4)
            self._rmse_knn         = round(rmse_knn, 4)
            self._trained_user_ids = set(df["user_id"].unique())

        logger.info(f"Pesos finais — SVD: {svd_w:.2f} | KNN: {1 - svd_w:.2f}")

    def predict(self, user_id: int, product_id: int) -> float:
        if self._svd is None:
            return 3.0
        p_svd = float(self._svd.predict(user_id, product_id).est)
        if self._knn is None:
            return p_svd
        p_knn = float(self._knn.predict(user_id, product_id).est)
        return self._svd_weight * p_svd + (1 - self._svd_weight) * p_knn

    def predict_batch(self, user_id: int, product_ids: list[int]) -> dict[int, float]:
        if self._svd is None:
            return {pid: 3.0 for pid in product_ids}
        results: dict[int, float] = {}
        for pid in product_ids:
            p_svd = float(self._svd.predict(user_id, pid).est)
            if self._knn is not None:
                p_knn = float(self._knn.predict(user_id, pid).est)
                results[pid] = self._svd_weight * p_svd + (1 - self._svd_weight) * p_knn
            else:
                results[pid] = p_svd
        return results

    def known_user(self, user_id: int) -> bool:
        return user_id in self._trained_user_ids

    @property
    def rmse_detail(self) -> dict:
        return {
            "svd":          self._rmse_svd,
            "knn_item_item": self._rmse_knn,
            "svd_weight":   round(self._svd_weight, 3),
        }


# Instância global — compartilhada entre routers
collaborative_filter = CollaborativeFilter()
