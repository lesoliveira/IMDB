#!/usr/bin/env python3

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Configurações
# -----------------------------
TRAIN_FILE = "train_st.txt"
TEST_FILE = "test_st.txt"

# -----------------------------
# Função para carregar dados
# -----------------------------
def load_data(file_path):
    X = []
    y = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Assume o formato: label v1,v2,v3...
                label, vector_str = line.split(" ", 1)
                vector = np.array(vector_str.split(","), dtype=float)
                
                y.append(int(label))
                X.append(vector)
        return np.array(X), np.array(y)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        exit(1)

# -----------------------------
# Carregar Treino e Teste
# -----------------------------
print("Carregando dados de treino...")
X_train, y_train = load_data(TRAIN_FILE)

print("Carregando dados de teste...")
X_test, y_test = load_data(TEST_FILE)

print(f"\nDataset carregado:")
print(f"Treino: {X_train.shape[0]} amostras | Dimensões: {X_train.shape[1]}")
print(f"Teste:  {X_test.shape[0]} amostras")

# -----------------------------
# Treinar LR
# -----------------------------
print("\nTreinando o modelo LR...")

lr = LogisticRegression() # explora melhor a linearidade do embedding
#lr = LinearSVC() # explora melhor a linearidade do embedding

lr.fit(X_train, y_train)

# -----------------------------
# Avaliação
# -----------------------------
y_pred = lr.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*30)
print("RESULTADOS")
print("="*30)
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))
