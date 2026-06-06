from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://fashion:fashion@db:5432/fashion_db"
    CSV_PATH: str = "/app/data/fashion_products_augmented.csv"
    FRONTEND_DIR: str = "/app/frontend"
    W_COLLABORATIVE:  float = 0.35
    W_CONTENT:        float = 0.25
    W_POPULARITY:     float = 0.20
    W_RATING:         float = 0.10
    W_BRAND:          float = 0.05
    W_SIZE:           float = 0.05

    class Config:
        env_file_encoding = "utf-8"


settings = Settings()
