import csv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------------
# Configurações
# -----------------------------
INPUT_FILE = input("Digite o nome do arquivo com os comentários: ")
OUTPUT_FILE = input("Digite o nome do arquivo de saída com as representações: ")
#MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_NAME = "all-mpnet-base-v2" # > tempo de processamento
BATCH_SIZE = 32

LABEL_MAP = {
    "pos": 1,
    "neg": -1
}

# -----------------------------
# Carregar modelo
# -----------------------------
model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# Ler CSV corretamente
# -----------------------------
labels = []
comments = []

with open(INPUT_FILE, "r", encoding="utf-8", errors="replace", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        review = row["review"].strip()
        label_str = row["label"].strip().lower()

        if not review or label_str not in LABEL_MAP:
            continue

        labels.append(LABEL_MAP[label_str])
        comments.append(review)

print(f"{len(comments)} comentários carregados.")

# -----------------------------
# Gerar embeddings com progress bar
# -----------------------------
embeddings = []

for i in tqdm(range(0, len(comments), BATCH_SIZE), desc="Gerando embeddings"):
    batch = comments[i:i + BATCH_SIZE]
    batch_embeddings = model.encode(
        batch,
        normalize_embeddings=True  # recomendado
    )
    embeddings.extend(batch_embeddings)

# -----------------------------
# Salvar rótulo + embedding
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for label, vector in zip(labels, embeddings):
        vector_str = ",".join(f"{v:.6f}" for v in vector)
        f.write(f"{label} {vector_str}\n")

print(f"Arquivo salvo em: {OUTPUT_FILE}")
