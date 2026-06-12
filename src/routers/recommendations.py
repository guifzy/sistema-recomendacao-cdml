from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.database import get_db
from src.models import Interaction, Product, User
from src.recommender.hybrid import evaluate_hybrid, recommend
from src.schemas import (
    HybridScoreResponse,
    InteractionCreate,
    InteractionOut,
    ProductCreate,
    ProductOut,
    RecommendationResponse,
    RecommendedProduct,
    UserCreate,
    UserOut,
    UserUpdate,
)

router = APIRouter()

@router.get("/products", response_model=list[ProductOut], tags=["Produtos"])
def list_products(
    skip:  int = Query(0,  ge=0),
    limit: int = Query(50, ge=1, le=500),
    db:    Session = Depends(get_db),
):
    """Lista todos os produtos com paginação."""
    return db.query(Product).offset(skip).limit(limit).all()


@router.get("/products/{product_id}", response_model=ProductOut, tags=["Produtos"])
def get_product(product_id: int, db: Session = Depends(get_db)):
    """Retorna um produto específico pelo ID."""
    p = db.get(Product, product_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Produto não encontrado.")
    return p


@router.post("/products", response_model=ProductOut, status_code=201, tags=["Produtos"])
def create_product(data: ProductCreate, db: Session = Depends(get_db)):
    """Adiciona um novo produto ao catálogo."""
    if db.get(Product, data.product_id):
        raise HTTPException(status_code=409, detail="Produto já existe.")
    product = Product(**data.model_dump(), avg_rating=0.0, interaction_count=0)
    db.add(product)
    db.commit()
    db.refresh(product)
    return product

@router.post("/users", response_model=UserOut, status_code=201, tags=["Usuários"])
def create_user(data: UserCreate, db: Session = Depends(get_db)):
    """Adiciona um novo usuário."""
    if db.get(User, data.user_id):
        raise HTTPException(status_code=409, detail="Usuário já existe.")
    user = User(**data.model_dump())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("/users/{user_id}", response_model=UserOut, tags=["Usuários"])
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Retorna um usuário pelo ID."""
    u = db.get(User, user_id)
    if u is None:
        raise HTTPException(status_code=404, detail="Usuário não encontrado.")
    return u


@router.put("/users/{user_id}/preferences", response_model=UserOut, tags=["Usuários"])
def update_preferences(user_id: int, data: UserUpdate, db: Session = Depends(get_db)):
    """Atualiza as preferências de um usuário."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Usuário não encontrado.")
    for field, value in data.model_dump(exclude_none=True).items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return user

@router.post("/interactions", response_model=InteractionOut, status_code=201, tags=["Interações"])
def add_interaction(data: InteractionCreate, db: Session = Depends(get_db)):
    """Registra a avaliação de um produto por um usuário."""
    user = db.get(User, data.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Usuário não encontrado.")
    product = db.get(Product, data.product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Produto não encontrado.")

    existing = (
        db.query(Interaction)
        .filter(Interaction.user_id == data.user_id, Interaction.product_id == data.product_id)
        .first()
    )
    if existing:
        # Atualiza rating existente
        existing.rating = data.rating
        db.commit()
        db.refresh(existing)
        intr = existing
    else:
        intr = Interaction(**data.model_dump())
        db.add(intr)
        db.commit()
        db.refresh(intr)

    # Atualiza avg_rating e interaction_count do produto
    all_ratings = (
        db.query(Interaction.rating)
        .filter(Interaction.product_id == data.product_id)
        .all()
    )
    ratings_list = [r.rating for r in all_ratings]
    product.avg_rating        = sum(ratings_list) / len(ratings_list)
    product.interaction_count = len(ratings_list)
    db.commit()

    return intr


@router.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recomendações"],
    summary="Sistema completo de recomendação híbrida",
)
def recommend_for_user(
    user_id:       int,
    top_n:         int     = Query(10, ge=1, le=50),
    exclude_seen:  bool    = Query(True),
    db:            Session = Depends(get_db),
):
    """
    Recomendações personalizadas utilizando o sistema híbrido completo
    (colaborativo + conteúdo + popularidade + avaliação + marca + tamanho).
    """
    if db.get(User, user_id) is None:
        raise HTTPException(status_code=404, detail="Usuário não encontrado.")

    items = recommend(user_id, db, top_n=top_n, exclude_seen=exclude_seen)
    return RecommendationResponse(
        user_id=user_id,
        method="hybrid",
        count=len(items),
        items=[RecommendedProduct(**i) for i in items],
    )


@router.get(
    "/top-rated",
    response_model=RecommendationResponse,
    tags=["Recomendações"],
    summary="Roupas mais bem avaliadas",
)
def top_rated(
    top_n:    int     = Query(10, ge=1, le=100),
    category: str | None = Query(None, description="Filtrar por categoria"),
    db:       Session = Depends(get_db),
):
    """Retorna os produtos com maior avaliação média (mínimo 5 interações)."""
    q = db.query(Product).filter(Product.interaction_count >= 5)
    if category:
        q = q.filter(Product.category == category)
    products = q.order_by(Product.avg_rating.desc()).limit(top_n).all()

    items = [
        RecommendedProduct(
            **{c.name: getattr(p, c.name) for c in p.__table__.columns},
            score=round((p.avg_rating - 1.0) / 4.0, 4),
            score_breakdown={"avg_rating": round(p.avg_rating, 3)},
        )
        for p in products
    ]
    return RecommendationResponse(user_id=0, method="top_rated", count=len(items), items=items)


@router.get(
    "/popular",
    response_model=RecommendationResponse,
    tags=["Recomendações"],
    summary="Roupas mais populares",
)
def popular(
    top_n:    int     = Query(10, ge=1, le=100),
    category: str | None = Query(None, description="Filtrar por categoria"),
    db:       Session = Depends(get_db),
):
    """Retorna os produtos com maior número de interações (mais populares)."""
    q = db.query(Product)
    if category:
        q = q.filter(Product.category == category)
    products = q.order_by(Product.interaction_count.desc()).limit(top_n).all()

    max_count = max((p.interaction_count for p in products), default=1) or 1
    items = [
        RecommendedProduct(
            **{c.name: getattr(p, c.name) for c in p.__table__.columns},
            score=round(p.interaction_count / max_count, 4),
            score_breakdown={"interaction_count": p.interaction_count},
        )
        for p in products
    ]
    return RecommendationResponse(user_id=0, method="popular", count=len(items), items=items)


@router.get(
    "/hybrid-score/{user_id}",
    response_model=HybridScoreResponse,
    tags=["Recomendações"],
    summary="Score de avaliação do sistema híbrido",
)
def hybrid_score(user_id: int, db: Session = Depends(get_db)):

    if db.get(User, user_id) is None:
        raise HTTPException(status_code=404, detail="Usuário não encontrado.")
    result  = evaluate_hybrid(user_id, db)
    top     = result.pop("top_recommendation")
    regime  = result.pop("regime", None)
    return HybridScoreResponse(
        **result,
        regime=regime,
        top_recommendation=RecommendedProduct(**top) if top else None,
    )
