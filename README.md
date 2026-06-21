# FashionAI — Sistema de Recomendação de Moda

Colaboradores

| Nome                         | Contribuição                                              |
| ---------------------------- | ----------------------------------------------------------- |
| **Guilherme Monteiro** | Arquitetura do sistema, motor híbrido adaptativo, backend  |
| **Lucas de Souza**     | Filtragem colaborativa (SVD + KNNBaseline)                  |
| **Leo Alec Marquez**   | Filtragem baseada em conteúdo (feature vectors ponderados) |
| **Italo Nascimento**   | Filtragem colaborativa (SVD + KNNBaseline)                  |
|**Caio Neves**| Front-end |

---

## Visão Geral

O FashionAI é um sistema completo de recomendação de produtos de moda que combina três abordagens complementares com **pesos adaptativos** de acordo com o histórico de cada usuário:

1. **Filtragem Colaborativa** — Combina SVD (fatores latentes globais) com KNNBaseline item-item (co-ocorrência local), com pesos entre os dois modelos calibrados automaticamente pelo RMSE de cada um no conjunto de teste.
2. **Filtragem Baseada em Conteúdo** — Representa cada produto como um vetor de features categóricas ponderadas por importância (marca, categoria, tipo, tamanho, cor, preço, avaliação) com similaridade cosseno, suportando aversão ativa a produtos mal avaliados.
3. **Sistema Híbrido Adaptativo** — Combina todos os sinais com pesos que variam de acordo com o regime do usuário: cold-start, warm-start ou denso.

---

## Arquitetura do Projeto

```
sis_ia_proj3/
├── data/
│   ├── fashion_products.csv               # Dataset original (1.000 linhas)
│   └── fashion_products_augmented.csv     # Dataset augmentado (~137k linhas)
├── src/
│   ├── main.py                            # Aplicação FastAPI principal
│   ├── config.py                          # Configurações e pesos do sistema
│   ├── database.py                        # Conexão SQLAlchemy + PostgreSQL
│   ├── models.py                          # Modelos ORM (Product, User, Interaction)
│   ├── schemas.py                         # Schemas Pydantic (request/response)
│   ├── init_db.py                         # Inicialização e carga do banco de dados
│   ├── recommender/
│   │   ├── collaborative.py               # SVD + KNNBaseline item-item (Surprise)
│   │   ├── content_based.py               # Feature vectors ponderados + cosine similarity
│   │   └── hybrid.py                      # Motor híbrido adaptativo + métricas
│   └── routers/
│       └── recommendations.py             # Todos os endpoints FastAPI
├── scripts/
│   └── augment_data.py                    # Script de augmentação (uso único)
├── frontend/
│   └── index.html                         # Interface web moderna (HTML/CSS/JS)
├── Dockerfile                             # Imagem do backend
├── docker-compose.yml                     # Orquestração dos serviços
└── requirements.txt                       # Dependências Python
```

---

## Pré-requisitos

- [Docker](https://www.docker.com/) >= 24.0
- [Docker Compose](https://docs.docker.com/compose/) >= 2.0
- Python 3.11+ *(apenas para o script de augmentação, que usa somente stdlib)*

---

## Como Executar

### 1. Gerar o Dataset Augmentado

O script expande o dataset original (1.000 linhas, 100 usuários) para aproximadamente **137.000 interações** com 5.000 usuários sintéticos. Usa apenas a stdlib do Python — nenhuma dependência extra necessária.

```bash
python scripts/augment_data.py
```

Saída esperada:

```
Carregando produtos originais...
Produtos carregados: 1000
Construindo perfis de usuários...
Gerando interações augmentadas...
  Processados 500/5000 usuários | Interações geradas: 13,973
  ...
Escrevendo 137,122 linhas em data/fashion_products_augmented.csv
Total de interações: 137,122
Usuários únicos:     5,000
Produtos únicos:     1,000
```

### 2. Subir os Containers

```bash
docker compose up --build
```

Isso iniciará dois serviços:

- **`db`** — PostgreSQL 16 na porta `5432`
- **`backend`** — FastAPI na porta `8000`

Na primeira execução, o backend automaticamente:

1. Aguarda o banco de dados ficar disponível (com retries automáticos)
2. Cria as tabelas via SQLAlchemy
3. Carrega o CSV augmentado em batches de 2.000 registros
4. Treina os modelos de recomendação em background (sem bloquear a API)

### 3. Acessar a Aplicação

| Serviço                         | URL                          |
| -------------------------------- | ---------------------------- |
| **Frontend**               | http://localhost:8000        |
| **Documentação Swagger** | http://localhost:8000/docs   |
| **Documentação ReDoc**   | http://localhost:8000/redoc  |
| **Health Check**           | http://localhost:8000/health |

O health check indica o status do treinamento e os RMSEs de cada modelo:

```json
{
  "status": "ok",
  "collaborative_trained": true,
  "content_trained": true,
  "cf_models": {
    "svd": 0.7821,
    "knn_item_item": 0.8134,
    "svd_weight": 0.510
  }
}
```

---

## Endpoints da API

### Recomendações

| Método | Endpoint                           | Descrição                                               |
| ------- | ---------------------------------- | --------------------------------------------------------- |
| `GET` | `/api/v1/recommend/{user_id}`    | **Sistema híbrido completo** com pesos adaptativos |
| `GET` | `/api/v1/top-rated`              | Produtos mais bem avaliados (mín. 5 interações)        |
| `GET` | `/api/v1/popular`                | Produtos mais populares por nº de interações           |
| `GET` | `/api/v1/hybrid-score/{user_id}` | Métricas de avaliação do sistema híbrido              |

### Produtos

| Método  | Endpoint                  | Descrição                |
| -------- | ------------------------- | -------------------------- |
| `GET`  | `/api/v1/products`      | Listar produtos (paginado) |
| `GET`  | `/api/v1/products/{id}` | Detalhes de um produto     |
| `POST` | `/api/v1/products`      | Adicionar novo produto     |

### Usuários

| Método  | Endpoint                           | Descrição             |
| -------- | ---------------------------------- | ----------------------- |
| `GET`  | `/api/v1/users/{id}`             | Detalhes de um usuário |
| `POST` | `/api/v1/users`                  | Adicionar novo usuário |
| `PUT`  | `/api/v1/users/{id}/preferences` | Atualizar preferências |

### Interações

| Método  | Endpoint                 | Descrição                      |
| -------- | ------------------------ | -------------------------------- |
| `POST` | `/api/v1/interactions` | Registrar avaliação de produto |

### Exemplos de uso

```bash
# Recomendações híbridas para o usuário 42
curl "http://localhost:8000/api/v1/recommend/42?top_n=10"

# Top 20 mais bem avaliados na categoria feminina
curl "http://localhost:8000/api/v1/top-rated?top_n=20&category=Women's Fashion"

# Produtos mais populares
curl "http://localhost:8000/api/v1/popular?top_n=10"

# Métricas do sistema para o usuário 42
curl "http://localhost:8000/api/v1/hybrid-score/42"

# Adicionar novo usuário
curl -X POST http://localhost:8000/api/v1/users \
  -H "Content-Type: application/json" \
  -d '{"user_id": 9999, "preferred_brand": "Nike", "preferred_size": "M"}'

# Atualizar preferências
curl -X PUT http://localhost:8000/api/v1/users/9999/preferences \
  -H "Content-Type: application/json" \
  -d '{"preferred_brand": "Adidas", "preferred_category": "Men'\''s Fashion"}'

# Registrar avaliação
curl -X POST http://localhost:8000/api/v1/interactions \
  -H "Content-Type: application/json" \
  -d '{"user_id": 9999, "product_id": 1, "rating": 4.5}'
```

---

## Modelo de Recomendação

### Dataset

O dataset `fashion_products.csv` contém interações usuário-produto com os seguintes atributos:

| Campo            | Tipo   | Descrição                                             |
| ---------------- | ------ | ------------------------------------------------------- |
| `User ID`      | int    | Identificador do usuário (1–100 no original)          |
| `Product ID`   | int    | Identificador do produto (1–1.000)                     |
| `Product Name` | string | Tipo do produto: Dress, Shoes, Jeans, T-shirt, Sweater  |
| `Brand`        | string | Marca: Zara, H&M, Gucci, Nike, Adidas                   |
| `Category`     | string | Segmento: Men's Fashion, Women's Fashion, Kids' Fashion |
| `Price`        | float  | Preço em R$ (10–100)                                  |
| `Rating`       | float  | Avaliação do usuário (1.0–5.0)                      |
| `Color`        | string | Cor: Black, White, Blue, Red, Green, Yellow             |
| `Size`         | string | Tamanho: S, M, L, XL                                    |

### Augmentação de Dados

O script `augment_data.py` gera ~137.000 interações realistas a partir dos 1.000 produtos originais:

- **5.000 usuários sintéticos** com perfis de preferência (marca, categoria e tamanho favoritos atribuídos aleatoriamente)
- **Distribuição log-normal** de interações por usuário — maioria com poucas interações, minoria com muitas (padrão realista de e-commerce)
- **Pesos de seleção de produto** — usuários interagem proporcionalmente mais com produtos da marca e categoria favoritas
- **Ratings com viés de perfil e ruído gaussiano** — usuários avaliam melhor produtos que correspondem às preferências, com desvio padrão de 0.5 para simular variabilidade humana

---

## Filtragem Colaborativa — SVD + KNNBaseline Item-Item

A filtragem colaborativa usa dois modelos da biblioteca `scikit-surprise` com pesos calibrados automaticamente pelo RMSE de cada um no conjunto de teste (15% dos dados).

### SVD (Singular Value Decomposition)

Fatora a matriz usuário-item em embeddings latentes que capturam preferências implícitas globais.

**Hiperparâmetros calibrados para este dataset:**

| Parâmetro    | Valor                   | Justificativa                                                                             |
| ------------- | ----------------------- | ----------------------------------------------------------------------------------------- |
| `n_factors` | `sqrt(n_items)` ≈ 32 | Heurística padrão da literatura; valores >100 causam overfitting em catálogos pequenos |
| `n_epochs`  | 30                      | Convergência estável com regularização adequada                                       |
| `reg_all`   | 0.05                    | Regularização maior que o padrão (0.02) para 1.000 itens                               |
| `biased`    | True                    | Inclui baseline por usuário e item, melhorando estimativas para usuários extremos       |

### KNNBaseline Item-Item

Calcula a similaridade de co-ocorrência entre itens: identifica quais produtos são frequentemente avaliados juntos por usuários similares. É especialmente eficaz neste dataset porque muitos produtos compartilham marca e categoria — o KNN captura padrões de co-compra que os fatores latentes do SVD não refletem diretamente.

| Parâmetro                | Valor                | Justificativa                                                  |
| ------------------------- | -------------------- | -------------------------------------------------------------- |
| `user_based`            | False                | Item-item: foca em co-ocorrências entre produtos              |
| `sim_options.name`      | `pearson_baseline` | Mais robusta que cosseno para pares com poucas co-avaliações |
| `sim_options.shrinkage` | 100                  | Penaliza estimativas baseadas em poucos dados em comum         |
| `k`                     | 40                   | Número de vizinhos mais similares considerados                |
| `min_k`                 | 3                    | Mínimo para predição; abaixo disso usa baseline global      |

**Calibração automática de pesos:** o modelo com menor RMSE no testset recebe automaticamente maior peso na predição final. Isso elimina a necessidade de ajuste manual dos pesos entre SVD e KNN.

---

## Filtragem Baseada em Conteúdo — Feature Vectors Ponderados

### Por que não TF-IDF

O vocabulário deste dataset tem apenas ~20 tokens únicos (5 marcas + 5 tipos + 3 categorias + 6 cores + 4 tamanhos). Com tantos produtos compartilhando os mesmos tokens, o IDF seria quase uniforme — todos os termos apareceriam em centenas de documentos, perdendo poder discriminativo. Para vocabulários categóricos pequenos, **one-hot encoding com pesos de importância** é a abordagem correta.

### Representação vetorial

Cada produto é representado por um vetor contínuo concatenando:

| Feature       | Representação | Peso          | Justificativa                                               |
| ------------- | --------------- | ------------- | ----------------------------------------------------------- |
| Marca         | One-hot (5 dim) | **2.0** | Principal discriminador de preferência em moda             |
| Categoria     | One-hot (3 dim) | **1.8** | Masculino/Feminino/Kids — filtro fundamental               |
| Tipo          | One-hot (5 dim) | **1.5** | Dress, Shoes, Jeans — preferência de estilo               |
| Tamanho       | One-hot (4 dim) | **1.2** | Crítico em moda: itens do tamanho errado são irrelevantes |
| Cor           | One-hot (6 dim) | **0.8** | Relevante, mas secundário à marca e tipo                  |
| Preço        | Contínuo [0,1] | **0.5** | Proxy de segmento de mercado                                |
| Rating médio | Contínuo [0,1] | **0.7** | Sinal de qualidade coletiva do produto                      |

Todos os vetores são normalizados L2 antes do cálculo de similaridade cosseno.

### Perfil do usuário com aversão ativa

O perfil vetorial do usuário é construído como soma ponderada pelos ratings **centralizados em 3.0**:

```python
w = (rating - 3.0) / 2.0
```

Resultado:

- Rating 5.0 → peso **+1.0** (o usuário ama este tipo de produto)
- Rating 3.0 → peso **0.0** (neutro, não influencia o perfil)
- Rating 1.0 → peso **−1.0** (o usuário ativamente evita este tipo de produto)

Essa estratégia de **aversão ativa** é superior à abordagem ingênua de usar apenas pesos positivos: o sistema aprende tanto preferências positivas quanto negativas do usuário, e o score de conteúdo pode ser negativo para produtos que conflitam com o perfil do usuário.

---

## Sistema Híbrido Adaptativo

### O problema dos pesos fixos

Um sistema com pesos fixos (CF=35%, sempre) tem um defeito fundamental: para um usuário novo com 2 interações, o CF recebe 35% de peso mas retorna essencialmente a média global — não tem dados suficientes para personalizar. O resultado são recomendações genéricas apresentadas como personalizadas.

### Pesos adaptativos por regime

O sistema detecta o regime de cada usuário e aplica pesos adequados ao volume de dados disponível:

| Regime               | Critério          | CF              | Conteúdo       | Popularidade    | Rating          | Marca           | Tamanho         |
| -------------------- | ------------------ | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| **Cold-start** | < 5 interações   | 10%             | 20%             | **40%**   | **20%**   | 5%              | 5%              |
| **Warm-start** | 5–30 interações | *interpolado* | *interpolado* | *interpolado* | *interpolado* | *interpolado* | *interpolado* |
| **Dense**      | > 30 interações  | **35%**   | **25%**   | 20%             | 10%             | 5%              | 5%              |

A transição warm-start usa interpolação linear suave entre cold e dense, evitando saltos bruscos no comportamento.

### Score final

```text
score = w_cf  × CF_norm
      + w_cb  × CB_norm          (scores negativos deslocados para [0, 2] antes de normalizar)
      + w_pop × (interações / máx_interações)
      + w_rat × (avg_rating − 1) / 4
      + w_br  × (1 se marca favorita, senão 0)
      + w_sz  × (1 se tamanho favorito, senão 0)
```

O response inclui `score_breakdown` com a contribuição individual de cada componente, e `regime` indica qual conjunto de pesos foi aplicado.

---

## Métricas de Avaliação

O endpoint `/api/v1/hybrid-score/{user_id}` retorna:

| Métrica               | Descrição                                                       |
| ---------------------- | ----------------------------------------------------------------- |
| `model_rmse`         | RMSE ponderado de SVD e KNNBaseline no conjunto de teste          |
| `coverage`           | Fração do catálogo já interagida pelo usuário                |
| `diversity_score`    | Razão de categorias únicas nas top-20 recomendações geradas   |
| `novelty_score`      | Proporção de itens recomendados ainda não vistos pelo usuário |
| `avg_rating_given`   | Rating médio histórico do usuário                              |
| `regime`             | Regime atual:`cold_start`, `warm_start` ou `dense`          |
| `weights_used`       | Pesos efetivamente aplicados nesta requisição                   |
| `top_recommendation` | Melhor produto recomendado com score e breakdown                  |

---

## Banco de Dados

### Esquema

```sql
products (
    product_id        INTEGER PRIMARY KEY,
    product_name      VARCHAR(100),
    brand             VARCHAR(100),
    category          VARCHAR(100),
    price             FLOAT,
    color             VARCHAR(50),
    size              VARCHAR(10),
    avg_rating        FLOAT,          -- atualizado a cada nova interação
    interaction_count INTEGER         -- atualizado a cada nova interação
)

users (
    user_id            INTEGER PRIMARY KEY,
    preferred_brand    VARCHAR(100),  -- marca mais frequente no histórico
    preferred_category VARCHAR(100),
    preferred_size     VARCHAR(10),
    created_at         TIMESTAMP
)

interactions (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER REFERENCES users(user_id),
    product_id INTEGER REFERENCES products(product_id),
    rating     FLOAT,
    created_at TIMESTAMP,
    UNIQUE (user_id, product_id)      -- no máximo 1 avaliação por par
)
```

---

## Configuração Avançada

### Variáveis de Ambiente

| Variável           | Padrão                                             | Descrição                       |
| ------------------- | --------------------------------------------------- | --------------------------------- |
| `DATABASE_URL`    | `postgresql://fashion:fashion@db:5432/fashion_db` | URL de conexão PostgreSQL        |
| `CSV_PATH`        | `/app/data/fashion_products_augmented.csv`        | Caminho do CSV augmentado         |
| `W_COLLABORATIVE` | `0.35`                                            | Peso CF no regime dense           |
| `W_CONTENT`       | `0.25`                                            | Peso conteúdo no regime dense    |
| `W_POPULARITY`    | `0.20`                                            | Peso popularidade no regime dense |
| `W_RATING`        | `0.10`                                            | Peso avaliação no regime dense  |
| `W_BRAND`         | `0.05`                                            | Peso bônus de marca              |
| `W_SIZE`          | `0.05`                                            | Peso bônus de tamanho            |

### Comandos úteis

```bash
# Subir em background
docker compose up --build -d

# Ver logs em tempo real
docker compose logs -f backend

# Parar sem apagar dados
docker compose down

# Parar e apagar o banco (reset completo)
docker compose down -v
```

---

## Decisões de Design

### Por que PostgreSQL e não SQLite?

Com ~137.000 interações e múltiplos índices (user_id, product_id, category, brand), o PostgreSQL oferece melhor desempenho em queries de agregação e concorrência. A separação em dois containers segue a arquitetura de microsserviços, tornando banco e aplicação independentemente gerenciáveis.

### Por que SVD + KNNBaseline e não só SVD?

Os dois modelos capturam sinais complementares. SVD aprende embeddings latentes globais (padrões de preferência entre usuários similares), enquanto o KNNBaseline item-item captura co-ocorrências locais (produtos frequentemente avaliados juntos). Para um catálogo com atributos repetidos, o KNN item-item complementa o SVD identificando padrões que os fatores latentes não refletem diretamente. Os pesos entre eles são calibrados automaticamente pelo RMSE de cada um.

### Por que feature vectors ponderados e não TF-IDF?

TF-IDF é projetado para texto livre com vocabulários ricos. Com ~20 tokens únicos neste dataset, o IDF seria quase uniforme e não diferenciaria os produtos. One-hot encoding com pesos explícitos por importância de feature é a abordagem correta para dados tabulares categóricos — é interpretável, eficiente computacionalmente e mais fiel ao domínio do problema.

### Por que pesos adaptativos no sistema híbrido?

Pesos fixos para CF são inadequados para usuários com poucas interações: o modelo retorna essencialmente a média global para usuários desconhecidos, contaminando o score híbrido. Pesos adaptativos garantem que cada componente receba peso proporcional à confiabilidade do seu sinal para aquele usuário específico.

---

## Tecnologias Utilizadas

| Tecnologia      | Versão | Uso                                          |
| --------------- | ------- | -------------------------------------------- |
| Python          | 3.11    | Linguagem principal                          |
| FastAPI         | 0.111   | Framework web / API REST                     |
| SQLAlchemy      | 2.0     | ORM / abstração de banco                   |
| PostgreSQL      | 16      | Banco de dados relacional                    |
| scikit-surprise | 1.1.4   | SVD e KNNBaseline item-item                  |
| scikit-learn    | 1.4     | Cosine similarity, MinMaxScaler              |
| pandas          | 2.2     | Manipulação de dados e feature engineering |
| numpy           | 1.26    | Operações vetoriais e matriciais           |
| Docker          | 24+     | Containerização                            |
| Docker Compose  | 2+      | Orquestração de serviços                  |

---

## Licença

Projeto acadêmico desenvolvido para a disciplina de Sistemas de IA.
