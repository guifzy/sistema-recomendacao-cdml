from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class ProductOut(BaseModel):
    product_id:        int
    product_name:      str
    brand:             str
    category:          str
    price:             float
    color:             str
    size:              str
    avg_rating:        float
    interaction_count: int

    model_config = {"from_attributes": True}


class ProductCreate(BaseModel):
    product_id:   int
    product_name: str
    brand:        str
    category:     str
    price:        float
    color:        str
    size:         str


class UserCreate(BaseModel):
    user_id:            int
    preferred_brand:    Optional[str] = None
    preferred_category: Optional[str] = None
    preferred_size:     Optional[str] = None


class UserUpdate(BaseModel):
    preferred_brand:    Optional[str] = None
    preferred_category: Optional[str] = None
    preferred_size:     Optional[str] = None


class UserOut(BaseModel):
    user_id:            int
    preferred_brand:    Optional[str]
    preferred_category: Optional[str]
    preferred_size:     Optional[str]

    model_config = {"from_attributes": True}


class InteractionCreate(BaseModel):
    user_id:    int
    product_id: int
    rating:     float = Field(..., ge=1.0, le=5.0)


class InteractionOut(BaseModel):
    id:         int
    user_id:    int
    product_id: int
    rating:     float

    model_config = {"from_attributes": True}


class RecommendedProduct(BaseModel):
    product_id:        int
    product_name:      str
    brand:             str
    category:          str
    price:             float
    color:             str
    size:              str
    avg_rating:        float
    interaction_count: int
    score:             float = Field(description="Score final do sistema híbrido (0-1)")
    score_breakdown:   Optional[dict] = Field(None, description="Detalhamento dos scores por componente")


class RecommendationResponse(BaseModel):
    user_id:      int
    method:       str
    count:        int
    items:        list[RecommendedProduct]


class HybridScoreResponse(BaseModel):
    user_id:                int
    total_interactions:     int
    coverage:               float = Field(description="% do catálogo coberto pelo usuário")
    avg_rating_given:       float
    model_rmse:             Optional[float]
    diversity_score:        float = Field(description="Diversidade das recomendações (0-1)")
    novelty_score:          float = Field(description="Novidade das recomendações (0-1)")
    weights_used:           dict
    regime:                 Optional[str] = Field(None, description="cold_start | warm_start | dense")
    top_recommendation:     Optional[RecommendedProduct]
