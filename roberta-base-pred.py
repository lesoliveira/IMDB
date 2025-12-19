import torch
import pandas as pd
import numpy as np

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Configurações
# -----------------------------
MODEL_DIR = "./roberta-imdb"
CSV_FILE = input("Digite o arquivo CSV para avaliação: ")
MAX_LENGTH = 512

LABEL_MAP = {"neg": 0, "pos": 1}
INV_LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Carregar modelo e tokenizer
# -----------------------------
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

print(f"Usando dispositivo: {DEVICE}")

# -----------------------------
# Carregar CSV
# -----------------------------
df = pd.read_csv(CSV_FILE)
df = df[["review", "label"]]
df["label"] = df["label"].map(LABEL_MAP)
df = df.dropna().reset_index(drop=True)

texts = df["review"].tolist()
y_true = df["label"].astype(int).tolist()

print(f"Total de comentários: {len(texts)}")

# -----------------------------
# Inferência em batch (eficiente)
# -----------------------------
BATCH_SIZE = 16
y_pred = []

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i + BATCH_SIZE]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

    y_pred.extend(preds.cpu().numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# -----------------------------
# Métricas
# -----------------------------
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n===== RESULTADOS =====")
print(f"Accuracy: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["NEGATIVE", "POSITIVE"],
    digits=4
))

print("Confusion Matrix:")
print(cm)

