import csv
import random
import math
import os

# script para gerar um dataset augmentado de interações usuário-produto
random.seed(42)

INPUT_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "fashion_products.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "fashion_products_augmented.csv")

TARGET_INTERACTIONS = 100_000
NUM_USERS           = 5_000   # usuários sintéticos

BRANDS     = ["Zara", "H&M", "Gucci", "Nike", "Adidas"]
CATEGORIES = ["Men's Fashion", "Women's Fashion", "Kids' Fashion"]
COLORS     = ["Black", "White", "Blue", "Red", "Green", "Yellow"]
SIZES      = ["S", "M", "L", "XL"]
NAMES      = ["Dress", "Shoes", "Jeans", "T-shirt", "Sweater"]

# Perfis de usuário: cada usuário tem marca e categoria favorita
USER_PROFILES: dict[int, dict] = {}

def build_user_profiles(n: int) -> None:
    for uid in range(1, n + 1):
        USER_PROFILES[uid] = {
            "fav_brand":    random.choice(BRANDS),
            "fav_category": random.choice(CATEGORIES),
            "fav_size":     random.choice(SIZES),
        }

def biased_rating(base_rating: float, user_id: int, product: dict) -> float:
    """Gera rating com viés de perfil de usuário + ruído gaussiano."""
    profile = USER_PROFILES.get(user_id, {})
    bonus = 0.0
    if profile.get("fav_brand") == product["Brand"]:
        bonus += 0.4
    if profile.get("fav_category") == product["Category"]:
        bonus += 0.3
    if profile.get("fav_size") == product["Size"]:
        bonus += 0.2
    noisy = base_rating + bonus + random.gauss(0, 0.5)
    return round(max(1.0, min(5.0, noisy)), 4)

def load_original_products(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main() -> None:
    print("Carregando produtos originais...")
    original_rows = load_original_products(INPUT_FILE)

    # Mapeia product_id → atributos do produto
    products: dict[int, dict] = {}
    for row in original_rows:
        pid = int(row["Product ID"])
        products[pid] = {
            "Product ID":   pid,
            "Product Name": row["Product Name"],
            "Brand":        row["Brand"],
            "Category":     row["Category"],
            "Price":        float(row["Price"]),
            "Rating":       float(row["Rating"]),
            "Color":        row["Color"],
            "Size":         row["Size"],
        }

    product_ids = list(products.keys())
    print(f"Produtos carregados: {len(product_ids)}")

    print("Construindo perfis de usuários...")
    build_user_profiles(NUM_USERS)
    # Número de interações por usuário segue distribuição log-normal
    # (maioria tem poucas interações, alguns têm muitas)
    interactions_per_user = TARGET_INTERACTIONS // NUM_USERS  # ~20
    extra = TARGET_INTERACTIONS - interactions_per_user * NUM_USERS

    rows_out: list[dict] = []
    seen: set[tuple[int, int]] = set()  # (user_id, product_id) para evitar duplicatas

    print("Gerando interações augmentadas...")
    for uid in range(1, NUM_USERS + 1):
        # Sorteia quantas interações este usuário terá
        n_interactions = max(1, int(random.lognormvariate(math.log(interactions_per_user), 0.8)))
        n_interactions = min(n_interactions, len(product_ids))

        profile = USER_PROFILES[uid]

        # Pesos de seleção de produto: produtos da categoria/marca favorita têm peso maior
        weights = []
        for pid in product_ids:
            p = products[pid]
            w = 1.0
            if p["Brand"] == profile["fav_brand"]:
                w *= 2.5
            if p["Category"] == profile["fav_category"]:
                w *= 2.0
            if p["Size"] == profile["fav_size"]:
                w *= 1.5
            weights.append(w)

        chosen = random.choices(product_ids, weights=weights, k=min(n_interactions * 3, len(product_ids)))

        added = 0
        for pid in chosen:
            if added >= n_interactions:
                break
            key = (uid, pid)
            if key in seen:
                continue
            seen.add(key)
            p = products[pid]
            rating = biased_rating(p["Rating"], uid, p)
            rows_out.append({
                "User ID":      uid,
                "Product ID":   pid,
                "Product Name": p["Product Name"],
                "Brand":        p["Brand"],
                "Category":     p["Category"],
                "Price":        p["Price"],
                "Rating":       rating,
                "Color":        p["Color"],
                "Size":         p["Size"],
            })
            added += 1

        if uid % 500 == 0:
            print(f"  Processados {uid}/{NUM_USERS} usuários | Interações geradas: {len(rows_out):,}")

    # Completa com interações aleatórias caso esteja abaixo do alvo
    while len(rows_out) < TARGET_INTERACTIONS:
        uid = random.randint(1, NUM_USERS)
        pid = random.choice(product_ids)
        key = (uid, pid)
        if key in seen:
            continue
        seen.add(key)
        p = products[pid]
        rating = biased_rating(p["Rating"], uid, p)
        rows_out.append({
            "User ID":      uid,
            "Product ID":   pid,
            "Product Name": p["Product Name"],
            "Brand":        p["Brand"],
            "Category":     p["Category"],
            "Price":        p["Price"],
            "Rating":       rating,
            "Color":        p["Color"],
            "Size":         p["Size"],
        })

    random.shuffle(rows_out)

    print(f"\nEscrevendo {len(rows_out):,} linhas em {OUTPUT_FILE} ...")
    fieldnames = ["User ID", "Product ID", "Product Name", "Brand", "Category", "Price", "Rating", "Color", "Size"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Concluído! Dataset augmentado salvo em: {OUTPUT_FILE}")
    print(f"Total de interações: {len(rows_out):,}")
    print(f"Usuários únicos:     {len(set(r['User ID'] for r in rows_out)):,}")
    print(f"Produtos únicos:     {len(set(r['Product ID'] for r in rows_out)):,}")


if __name__ == "__main__":
    main()
