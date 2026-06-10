"""
Sistema de Recomendação Híbrido.

Estratégia adaptativa por nível de interação do usuário:

  COLD-START  (< 5 interações):
    O CF não tem dados suficientes para o usuário → cai para popularidade e
    avaliação (sinal global). Conteúdo usa as poucas interações disponíveis.
    Pesos: popularidade 40%, avaliação 30%, conteúdo 20%, CF 10%.

  WARM-START  (5–30 interações):
    CF começa a ter sinal. Transição gradual para o CF ser mais relevante.

  DENSE       (> 30 interações):
    CF é o sinal mais forte. Pesos padrão das configurações são usados.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from src.config import settings
from src.models import Interaction, Product, User
from src.recommender.collaborative import collaborative_filter
from src.recommender.content_based import content_filter

logger = logging.getLogger(__name__)


def _normalize(values: dict[int, float]) -> dict[int, float]:
    """Min-max normalization preservando proporções."""
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    spread = vmax - vmin
    if spread < 1e-9:
        return {k: 0.5 for k in values}
    return {k: (v - vmin) / spread for k, v in values.items()}


def _get_user_history(user_id: int, db: Session) -> list[tuple[int, float]]:
    rows = (
        db.query(Interaction.product_id, Interaction.rating)
        .filter(Interaction.user_id == user_id)
        .all()
    )
    return [(int(r.product_id), float(r.rating)) for r in rows]


def _get_user_preferences(user_id: int, db: Session) -> dict:
    user = db.get(User, user_id)
    if user is None:
        return {}
    return {
        "preferred_brand":    user.preferred_brand,
        "preferred_category": user.preferred_category,
        "preferred_size":     user.preferred_size,
    }


def _adaptive_weights(n_interactions: int) -> dict[str, float]:

    if n_interactions < 5:
        # Cold-start: sem dados para CF, confia em sinais globais
        return {
            "collaborative": 0.10,
            "content":       0.20,
            "popularity":    0.40,
            "rating":        0.20,
            "brand":         0.05,
            "size":          0.05,
        }
    if n_interactions < 30:
        # Warm-start: interpolação linear entre cold e dense
        t = (n_interactions - 5) / 25.0  # 0→1
        cold  = _adaptive_weights(0)
        dense = {
            "collaborative": settings.W_COLLABORATIVE,
            "content":       settings.W_CONTENT,
            "popularity":    settings.W_POPULARITY,
            "rating":        settings.W_RATING,
            "brand":         settings.W_BRAND,
            "size":          settings.W_SIZE,
        }
        return {k: cold[k] * (1 - t) + dense[k] * t for k in cold}

    # Dense: usa os pesos das configurações
    return {
        "collaborative": settings.W_COLLABORATIVE,
        "content":       settings.W_CONTENT,
        "popularity":    settings.W_POPULARITY,
        "rating":        settings.W_RATING,
        "brand":         settings.W_BRAND,
        "size":          settings.W_SIZE,
    }


def recommend(
    user_id:      int,
    db:           Session,
    top_n:        int  = 10,
    exclude_seen: bool = True,
) -> list[dict]:
   
    history   = _get_user_history(user_id, db)
    seen_ids  = {pid for pid, _ in history}
    prefs     = _get_user_preferences(user_id, db)
    weights   = _adaptive_weights(len(history))

    all_products: list[Product] = db.query(Product).all()
    if not all_products:
        return []

    candidates = [
        p for p in all_products
        if not (exclude_seen and p.product_id in seen_ids)
    ] or list(all_products)

    candidate_ids    = [p.product_id for p in candidates]
    max_interactions = max((p.interaction_count for p in all_products), default=1) or 1

    cf_raw   = collaborative_filter.predict_batch(user_id, candidate_ids)
    cf_norm  = _normalize(cf_raw)

    cb_raw   = content_filter.score_for_user(history, candidate_ids)
    # Conteúdo pode ter scores negativos (produto que o usuário odeia)
    # Deslocamos para [0, 1] antes de normalizar
    cb_shifted = {pid: v + 1.0 for pid, v in cb_raw.items()}  # [-1,1] → [0,2]
    cb_norm    = _normalize(cb_shifted)

    fav_brand = prefs.get("preferred_brand")
    fav_size  = prefs.get("preferred_size")

    results = []
    for p in candidates:
        pid = p.product_id

        cf_s  = cf_norm.get(pid, 0.5)
        cb_s  = cb_norm.get(pid, 0.5)
        pop_s = p.interaction_count / max_interactions
        rat_s = max(0.0, (p.avg_rating - 1.0) / 4.0)
        bra_s = 1.0 if (fav_brand and p.brand == fav_brand) else 0.0
        siz_s = 1.0 if (fav_size  and p.size  == fav_size)  else 0.0

        score = (
            weights["collaborative"] * cf_s
            + weights["content"]     * cb_s
            + weights["popularity"]  * pop_s
            + weights["rating"]      * rat_s
            + weights["brand"]       * bra_s
            + weights["size"]        * siz_s
        )

        results.append({
            "product_id":        p.product_id,
            "product_name":      p.product_name,
            "brand":             p.brand,
            "category":          p.category,
            "price":             p.price,
            "color":             p.color,
            "size":              p.size,
            "avg_rating":        round(p.avg_rating, 3),
            "interaction_count": p.interaction_count,
            "score":             round(score, 4),
            "score_breakdown": {
                "collaborative":  round(cf_s,  4),
                "content_based":  round(cb_s,  4),
                "popularity":     round(pop_s, 4),
                "avg_rating":     round(rat_s, 4),
                "brand_match":    round(bra_s, 4),
                "size_match":     round(siz_s, 4),
            },
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def evaluate_hybrid(user_id: int, db: Session) -> dict:
    """Calcula métricas qualitativas do sistema híbrido para um usuário."""
    history   = _get_user_history(user_id, db)
    all_count = db.query(Product).count()
    weights   = _adaptive_weights(len(history))

    coverage         = len(history) / all_count if all_count else 0.0
    avg_rating_given = float(np.mean([r for _, r in history])) if history else 0.0

    recs = recommend(user_id, db, top_n=20, exclude_seen=True)

    if recs:
        categories = set(r["category"] for r in recs)
        diversity  = len(categories) / len(recs)
        seen_pids  = {pid for pid, _ in history}
        novelty    = sum(1 for r in recs if r["product_id"] not in seen_pids) / len(recs)
        top_rec    = recs[0]
    else:
        diversity = 0.0
        novelty   = 0.0
        top_rec   = None

    regime = (
        "cold_start" if len(history) < 5
        else "warm_start" if len(history) < 30
        else "dense"
    )

    return {
        "user_id":            user_id,
        "total_interactions": len(history),
        "coverage":           round(coverage, 4),
        "avg_rating_given":   round(avg_rating_given, 3),
        "model_rmse":         collaborative_filter.rmse,
        "diversity_score":    round(diversity, 4),
        "novelty_score":      round(novelty, 4),
        "weights_used":       {k: round(v, 3) for k, v in weights.items()},
        "regime":             regime,
        "top_recommendation": top_rec,
    }
