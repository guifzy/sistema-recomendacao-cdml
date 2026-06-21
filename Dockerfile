FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema necessárias para compilar extensões C (scikit-surprise)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código fonte e arquivos de suporte
COPY src/       ./src/
COPY frontend/  ./frontend/
COPY scripts/   ./scripts/
COPY data/      ./data/

# Gera o dataset augmentado caso não esteja presente
# (normalmente já está no repositório, mas garante reprodutibilidade)
RUN if [ ! -f ./data/fashion_products_augmented.csv ]; then \
      echo "CSV augmentado não encontrado — gerando agora..."; \
      python scripts/augment_data.py; \
    else \
      echo "CSV augmentado já presente."; \
    fi

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
