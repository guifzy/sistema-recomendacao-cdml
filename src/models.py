from datetime import datetime

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, ForeignKey, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from src.database import Base


class Product(Base):
    __tablename__ = "products"

    product_id       = Column(Integer, primary_key=True, index=True)
    product_name     = Column(String(100), nullable=False)
    brand            = Column(String(100), nullable=False, index=True)
    category         = Column(String(100), nullable=False, index=True)
    price            = Column(Float, nullable=False)
    color            = Column(String(50), nullable=False)
    size             = Column(String(10), nullable=False)
    avg_rating       = Column(Float, default=0.0)
    interaction_count = Column(Integer, default=0)

    interactions = relationship("Interaction", back_populates="product")


class User(Base):
    __tablename__ = "users"

    user_id           = Column(Integer, primary_key=True, index=True)
    preferred_brand   = Column(String(100), nullable=True)
    preferred_category = Column(String(100), nullable=True)
    preferred_size    = Column(String(10),  nullable=True)
    created_at        = Column(DateTime, default=datetime.utcnow)

    interactions = relationship("Interaction", back_populates="user")


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (UniqueConstraint("user_id", "product_id", name="uq_user_product"),)

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("users.user_id"),    nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.product_id"), nullable=False, index=True)
    rating     = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user    = relationship("User",    back_populates="interactions")
    product = relationship("Product", back_populates="interactions")
