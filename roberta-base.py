import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -----------------------------
# Configurações
# -----------------------------
TRAIN_FILE = input("Digite o arquivo CSV de TREINO: ")
TEST_FILE = input("Digite o arquivo CSV de TESTE: ")

MODEL_NAME = "roberta-base"
MAX_LENGTH = 512
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# -----------------------------
# Função para carregar CSV
# -----------------------------
def load_csv(path):
    df = pd.read_csv(path)
    df = df[["review", "label"]]

    label_map = {"neg": 0, "pos": 1}
    df["label"] = df["label"].map(label_map)

    df = df.dropna().reset_index(drop=True)
    return df

train_df = load_csv(TRAIN_FILE)
test_df = load_csv(TEST_FILE)

print(f"Treino: {len(train_df)} | Teste: {len(test_df)}")

# -----------------------------
# Converter para HuggingFace Dataset
# -----------------------------
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# -----------------------------
# Tokenizer e modelo
# -----------------------------
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# -----------------------------
# Tokenização
# -----------------------------
def tokenize(batch):
    return tokenizer(
        batch["review"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Remover texto cru
train_ds = train_ds.remove_columns(["review"])
test_ds = test_ds.remove_columns(["review"])

# Renomear label
train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

train_ds.set_format("torch")
test_ds.set_format("torch")

# -----------------------------
# Métricas
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# -----------------------------
# Argumentos de treinamento
# -----------------------------
training_args = TrainingArguments(
    output_dir="./roberta-imdb",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

# -----------------------------
# Treinamento
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

trainer.train()

# -----------------------------
# Avaliação final (SOMENTE teste)
# -----------------------------
results = trainer.evaluate()

trainer.save_model("./roberta-imdb")
tokenizer.save_pretrained("./roberta-imdb")

print("\n===== RESULTADOS NO TESTE =====")
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"F1-score: {results['eval_f1']:.4f}")
