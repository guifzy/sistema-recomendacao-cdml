from __future__ import annotations

import logging
import threading

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import settings
from src.init_db import init_db
from src.database import SessionLocal
from src.models import Interaction
from src.recommender.collaborative import collaborative_filter
from src.recommender.content_based import content_filter
from src.routers.recommendations import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _train_models() -> None:
    """Treina os modelos em background após a inicialização do DB."""
    logger.info("Iniciando treinamento dos modelos de recomendação...")
    db = SessionLocal()
    try:
        # Carrega interações para Filtragem Colaborativa
        rows = db.query(
            Interaction.user_id, Interaction.product_id, Interaction.rating
        ).all()
        df_interactions = pd.DataFrame(rows, columns=["user_id", "product_id", "rating"])

        if len(df_interactions) < 10:
            logger.warning("Dados insuficientes para treinar modelos.")
            return

        collaborative_filter.train(df_interactions)

        # Carrega produtos para Filtragem Baseada em Conteúdo
        from src.models import Product
        products = db.query(Product).all()
        df_products = pd.DataFrame(
            [
                {
                    "product_id":   p.product_id,
                    "product_name": p.product_name,
                    "brand":        p.brand,
                    "category":     p.category,
                    "price":        p.price,
                    "color":        p.color,
                    "size":         p.size,
                    "avg_rating":   p.avg_rating,
                }
                for p in products
            ]
        )
        content_filter.train(df_products)
        logger.info("Modelos treinados com sucesso.")
    except Exception as exc:
        logger.error(f"Erro ao treinar modelos: {exc}", exc_info=True)
    finally:
        db.close()


app = FastAPI(
    title="Fashion Recommendation API",
    description=(
        "Sistema de recomendação híbrido de moda.\n\n"
        "Combina filtragem colaborativa (SVD + KNNBaseline item-item), "
        "filtragem baseada em conteúdo (feature vectors ponderados) "
        "e sinais de popularidade e avaliação com pesos adaptativos por nível de histórico do usuário."
    ),
    version="1.0.0",
    contact={
        "name": "Guilherme Monteiro, Lucas de Souza, Leo Alec Marquez, Italo Nascimento",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# Serve o frontend estático (caminho configurável via FRONTEND_DIR)
import os as _os
_frontend = settings.FRONTEND_DIR
if _os.path.isdir(_frontend):
    app.mount("/static", StaticFiles(directory=_frontend), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(_os.path.join(_frontend, "index.html"))


@app.on_event("startup")
def startup_event():
    logger.info("Iniciando aplicação...")
    init_db()
    t = threading.Thread(target=_train_models, daemon=True)
    t.start()
    logger.info("Aplicação iniciada. Treinamento dos modelos em andamento (background).")


@app.get("/health", tags=["Sistema"])
def health():
    return {
        "status":                "ok",
        "collaborative_trained": collaborative_filter.is_trained,
        "content_trained":       content_filter.is_trained,
        "cf_models":             collaborative_filter.rmse_detail if collaborative_filter.is_trained else None,
    }
