from __future__ import annotations

import csv
import logging
import os
import time
from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from src.config import settings
from src.database import Base, engine, SessionLocal
from src.models import Interaction, Product, User

logger = logging.getLogger(__name__)


def wait_for_db(retries: int = 15, delay: int = 3) -> None:
    # SQLite é um arquivo local — não precisa aguardar conexão TCP
    if settings.DATABASE_URL.startswith("sqlite"):
        logger.info("SQLite detectado — pulando wait_for_db.")
        return
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Banco de dados disponível.")
            return
        except OperationalError:
            logger.warning(f"Banco não disponível. Tentativa {attempt}/{retries} — aguardando {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Não foi possível conectar ao banco de dados.")


def load_csv(path: str) -> tuple[dict, dict, list]:
    """Lê o CSV e retorna (products, users, interactions)."""
    products: dict[int, dict]  = {}
    users: dict[int, dict]     = {}
    interactions: list[dict]   = []

    # Acumuladores para calcular avg_rating e interaction_count
    rating_sums:   defaultdict[int, float] = defaultdict(float)
    rating_counts: defaultdict[int, int]   = defaultdict(int)
    user_brand_counts: defaultdict[int, defaultdict] = defaultdict(lambda: defaultdict(int))
    user_cat_counts:   defaultdict[int, defaultdict] = defaultdict(lambda: defaultdict(int))
    user_size_counts:  defaultdict[int, defaultdict] = defaultdict(lambda: defaultdict(int))

    logger.info(f"Lendo CSV: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["Product ID"])
            uid = int(row["User ID"])
            rat = float(row["Rating"])

            if pid not in products:
                products[pid] = {
                    "product_id":   pid,
                    "product_name": row["Product Name"],
                    "brand":        row["Brand"],
                    "category":     row["Category"],
                    "price":        float(row["Price"]),
                    "color":        row["Color"],
                    "size":         row["Size"],
                }
            rating_sums[pid]   += rat
            rating_counts[pid] += 1

            if uid not in users:
                users[uid] = {"user_id": uid}
            user_brand_counts[uid][row["Brand"]]    += 1
            user_cat_counts[uid][row["Category"]]   += 1
            user_size_counts[uid][row["Size"]]       += 1

            interactions.append({"user_id": uid, "product_id": pid, "rating": rat})

    # Completar campos agregados nos produtos
    for pid, p in products.items():
        p["avg_rating"]        = rating_sums[pid] / rating_counts[pid]
        p["interaction_count"] = rating_counts[pid]

    # Preferências dos usuários (moda de cada atributo)
    for uid, u in users.items():
        u["preferred_brand"]    = max(user_brand_counts[uid], key=user_brand_counts[uid].get)
        u["preferred_category"] = max(user_cat_counts[uid],   key=user_cat_counts[uid].get)
        u["preferred_size"]     = max(user_size_counts[uid],  key=user_size_counts[uid].get)

    logger.info(f"CSV carregado | Produtos: {len(products)} | Usuários: {len(users)} | Interações: {len(interactions)}")
    return products, users, interactions


def populate_db(products: dict, users: dict, interactions: list, batch_size: int = 2000) -> None:
    db = SessionLocal()
    try:
        # Produtos
        logger.info("Inserindo produtos...")
        product_objs = [Product(**p) for p in products.values()]
        for i in range(0, len(product_objs), batch_size):
            db.bulk_save_objects(product_objs[i : i + batch_size])
            db.commit()

        # Usuários
        logger.info("Inserindo usuários...")
        user_objs = [User(**u) for u in users.values()]
        for i in range(0, len(user_objs), batch_size):
            db.bulk_save_objects(user_objs[i : i + batch_size])
            db.commit()

        # Interações (deduplicadas)
        logger.info("Inserindo interações...")
        seen: set[tuple] = set()
        interaction_objs = []
        for intr in interactions:
            key = (intr["user_id"], intr["product_id"])
            if key not in seen:
                seen.add(key)
                interaction_objs.append(Interaction(**intr))

        for i in range(0, len(interaction_objs), batch_size):
            db.bulk_save_objects(interaction_objs[i : i + batch_size])
            db.commit()

        logger.info("Banco de dados populado com sucesso.")
    finally:
        db.close()


def init_db() -> None:
    wait_for_db()
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        count = db.query(Product).count()
    finally:
        db.close()

    if count > 0:
        logger.info(f"Banco já populado ({count} produtos). Pulando carga inicial.")
        return

    if not os.path.exists(settings.CSV_PATH):
        raise FileNotFoundError(
            f"CSV augmentado não encontrado: {settings.CSV_PATH}\n"
            "Execute primeiro: python scripts/augment_data.py"
        )

    products, users, interactions = load_csv(settings.CSV_PATH)
    populate_db(products, users, interactions)
