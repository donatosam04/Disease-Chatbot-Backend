import numpy as np
import pandas as pd
import torch
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

# ======================================================
# CONFIG  (IMPORTANT: uses AUGMENTED dataset)
# ======================================================
DATA = "data/processed/respiratory10_train_realistic.csv"
OUT  = Path("models_saved/intent_respiratory10")
MODEL = "bert-base-uncased"

OUT.mkdir(parents=True, exist_ok=True)

# ======================================================
# GPU CHECK
# ======================================================
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(DATA)
assert {"text", "label"}.issubset(df.columns), "CSV must contain text + label columns"

le = LabelEncoder()
df["labels"] = le.fit_transform(df["label"])

num_labels = len(le.classes_)
print("Number of labels:", num_labels)

# ======================================================
# CLASS WEIGHTS  ⭐⭐ STEP-1 FIX ⭐⭐
# ======================================================
label_counts = df["labels"].value_counts().sort_index()
class_weights = 1.0 / label_counts
class_weights = class_weights / class_weights.sum()
class_weights = torch.tensor(class_weights.values, dtype=torch.float)

print("Class weights:", class_weights.tolist())

# ======================================================
# TOKENIZATION
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

dataset = Dataset.from_pandas(df[["text", "labels"]])
dataset = dataset.map(tokenize, batched=True)

split = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = split["train"].with_format("torch")
val_ds   = split["test"].with_format("torch")

print("Train samples:", len(train_ds))
print("Validation samples:", len(val_ds))

# ======================================================
# MODEL
# ======================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=num_labels
)

# ======================================================
# CUSTOM TRAINER (weighted loss)
# ======================================================
class WeightedTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,  # 🔑 IMPORTANT FIX
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ======================================================
# METRICS
# ======================================================
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "macro_f1": f1_score(
            p.label_ids,
            preds,
            average="macro",
            zero_division=0
        ),
    }

# ======================================================
# TRAINING ARGUMENTS
# ======================================================
args = TrainingArguments(
    output_dir=str(OUT),

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    num_train_epochs=8,
    learning_rate=2e-5,
    weight_decay=0.01,

    logging_steps=50,
    save_steps=500,
    save_total_limit=2,

    fp16=torch.cuda.is_available(),
    report_to=[],   # no wandb / tensorboard
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# ======================================================
# TRAIN
# ======================================================
trainer.train()

# ======================================================
# SAVE MODEL + TOKENIZER + LABEL MAP
# ======================================================
trainer.save_model(OUT)
tokenizer.save_pretrained(OUT)

with open(OUT / "labels.txt", "w", encoding="utf-8") as f:
    for i, lbl in enumerate(le.classes_):
        f.write(f"{i}\t{lbl}\n")

print("✅ respiratory10 model training complete (weighted loss)")
