import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

# =====================
# CONFIG
# =====================
DATA = "data/processed/respiratory5_bert.csv"
OUT  = Path("models_saved/intent_respiratory5")
MODEL = "bert-base-uncased"

OUT.mkdir(parents=True, exist_ok=True)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

# =====================
# METRICS
# =====================
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "macro_f1": f1_score(p.label_ids, preds, average="macro", zero_division=0)
    }

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(DATA)

le = LabelEncoder()
df["labels"] = le.fit_transform(df["label"])

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = Dataset.from_pandas(df[["text", "labels"]])
dataset = dataset.map(tokenize, batched=True)

split = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = split["train"].with_format("torch")
val_ds   = split["test"].with_format("torch")

print("Train samples:", len(train_ds))
print("Validation samples:", len(val_ds))

# =====================
# MODEL
# =====================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=len(le.classes_)
)

# =====================
# TRAINING ARGUMENTS
# ⚠️ ABSOLUTELY NO evaluation_strategy HERE
# =====================
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
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# =====================
# TRAIN
# =====================
trainer.train()

# =====================
# SAVE
# =====================
trainer.save_model(OUT)
tokenizer.save_pretrained(OUT)

with open(OUT / "labels.txt", "w") as f:
    for i, lbl in enumerate(le.classes_):
        f.write(f"{i}\t{lbl}\n")

print("✅ respiratory5 model training complete")
