#!/usr/bin/env python3

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from tqdm import tqdm

# -----------------------------
# Configurações
# -----------------------------
TRAIN_FILE = input("Arquivo CSV de TREINO: ")
TEST_FILE = input("Arquivo CSV de TESTE: ")

OUTPUT_TRAIN = "train_w2v.txt"
OUTPUT_TEST = "test_w2v.txt"

VECTOR_SIZE = 300
WINDOW = 5
MIN_COUNT = 5
WORKERS = 4

LABEL_MAP = {"pos": 1, "neg": -1}

# -----------------------------
# Função para gerar vetor médio
# -----------------------------
def document_vector(doc, model):
	words = [w for w in doc if w in model.wv]
	if len(words) == 0:
		return np.zeros(model.vector_size)
	return np.mean(model.wv[words], axis=0)

# -----------------------------
# Ler CSV
# -----------------------------
def load_data(csv_file):
	df = pd.read_csv(csv_file)
	df = df[["review", "label"]]
	df["label"] = df["label"].map(LABEL_MAP)
	df = df.dropna()
	return df

train_df = load_data(TRAIN_FILE)
test_df = load_data(TEST_FILE)

print(f"Treino: {len(train_df)} | Teste: {len(test_df)}")

# -----------------------------
# Tokenização
# -----------------------------
train_tokens = [simple_preprocess(text) for text in train_df["review"]]
test_tokens = [simple_preprocess(text) for text in test_df["review"]]

# -----------------------------
# Treinar Word2Vec (SOMENTE treino)
# -----------------------------
print("Treinando Word2Vec...")
w2v_model = Word2Vec(
	sentences=train_tokens,
	vector_size=VECTOR_SIZE,
	window=WINDOW,
	min_count=MIN_COUNT,
	workers=WORKERS,
	epochs=10
	
)

# -----------------------------
# Gerar embeddings
# -----------------------------
print("Gerando embeddings de treino...")
X_train = [
	document_vector(doc, w2v_model)
	for doc in tqdm(train_tokens)
]

print("Gerando embeddings de teste...")
X_test = [
	document_vector(doc, w2v_model)
	for doc in tqdm(test_tokens)
]

# -----------------------------
# Salvar arquivos
# -----------------------------
def save_embeddings(filename, labels, vectors):
	with open(filename, "w", encoding="utf-8") as f:
		for label, vec in zip(labels, vectors):
			vec_str = ",".join(f"{v:.6f}" for v in vec)
			f.write(f"{label} {vec_str}\n")
			
save_embeddings(OUTPUT_TRAIN, train_df["label"], X_train)
save_embeddings(OUTPUT_TEST, test_df["label"], X_test)

print("Arquivos gerados:")
print(f"- {OUTPUT_TRAIN}")
print(f"- {OUTPUT_TEST}")
