import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path


TRAIN = "data/processed/respiratory7_train.csv"
VAL   = "data/processed/respiratory7_val.csv"
OUT   = Path("models_saved/intent_respiratory7")
MODEL = "bert-base-uncased"

OUT.mkdir(parents=True, exist_ok=True)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "macro_f1": f1_score(p.label_ids, preds, average="macro")
    }


df_tr = pd.read_csv(TRAIN)
df_va = pd.read_csv(VAL)

le = LabelEncoder()
df_tr["y"] = le.fit_transform(df_tr["label"])
df_va["y"] = le.transform(df_va["label"])


tok = AutoTokenizer.from_pretrained(MODEL)

def tok_fn(batch):
    return tok(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tr_ds = Dataset.from_pandas(df_tr[["text", "y"]]).map(tok_fn, batched=True)
va_ds = Dataset.from_pandas(df_va[["text", "y"]]).map(tok_fn, batched=True)

tr_ds = tr_ds.rename_column("y", "labels").with_format("torch")
va_ds = va_ds.rename_column("y", "labels").with_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=len(le.classes_)
)

args = TrainingArguments(
    output_dir=str(OUT),

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,

    logging_steps=10,
    save_steps=50,
    eval_steps=50,
    save_total_limit=2,

    do_eval=True,     # legacy-compatible evaluation
    fp16=False,       # CPU-safe
    report_to=[]
)

# =========================
# Trainer (NO EarlyStopping)
# =========================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tr_ds,
    eval_dataset=va_ds,
    tokenizer=tok,
    compute_metrics=compute_metrics
)

# =========================
# Train & save
# =========================
trainer.train()

trainer.save_model(OUT)
tok.save_pretrained(OUT)

with open(OUT / "labels.txt", "w") as f:
    for i, l in enumerate(le.classes_):
        f.write(f"{i}\t{l}\n")
