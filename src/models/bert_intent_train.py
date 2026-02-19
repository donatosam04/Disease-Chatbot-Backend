import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from pathlib import Path

TRAIN = "data/processed/bert_intent_train_reduced.csv"
VAL = "data/processed/bert_intent_val_reduced.csv"
OUT_DIR = Path("models_saved/intent")
MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-uncased")
MAX_LEN = 128

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load splits
    tr = pd.read_csv(TRAIN)
    va = pd.read_csv(VAL)

    # Encode labels on train
    le = LabelEncoder()
    tr["y"] = le.fit_transform(tr["label"])

    # Align/encode validation
    va = va[va["label"].isin(le.classes_)].reset_index(drop=True)
    if "y" not in va.columns and len(va) > 0:
        va["y"] = le.transform(va["label"])
    if len(va) == 0:
        va = tr.sample(n=min(800, len(tr)//4), random_state=42)[["text", "label", "y"]].copy()
        print("Validation was empty after filtering; sampled from train for eval only.")

    print(f"Train rows: {len(tr)} | Val rows: {len(va)} | Classes: {len(le.classes_)}")

    # Processor/tokenizer (fix deprecation)
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
    except Exception:
        pass
    tok = processor if processor is not None else AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokf(batch):
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    tr_ds = Dataset.from_pandas(tr[["text", "y"]]).map(tokf, batched=True)
    tr_ds = tr_ds.rename_column("y", "labels")
    va_ds = Dataset.from_pandas(va[["text", "y"]]).map(tokf, batched=True)
    va_ds = va_ds.rename_column("y", "labels")

    cols = ["input_ids", "attention_mask", "labels"]
    tr_ds.set_format("torch", columns=cols)
    va_ds.set_format("torch", columns=cols)

    # Model config and init (classifier head will be freshly initialized by HF)
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(le.classes_),
        id2label=dict(enumerate(le.classes_)),
        label2id={l: i for i, l in dict(enumerate(le.classes_)).items()},
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # Batch size selection
    per_device_bs = 16 if device == "cuda" else 8

    # Training arguments: compatible with older transformers
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        overwrite_output_dir=True,
        learning_rate=5e-5,
        per_device_train_batch_size=8 if device == "cpu" else 16,
        per_device_eval_batch_size=8 if device == "cpu" else 16,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="linear" if hasattr(TrainingArguments, "lr_scheduler_type") else None,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to=[],
        dataloader_num_workers=0 if device == "cpu" else 2,
        dataloader_pin_memory=torch.cuda.is_available(),
        max_grad_norm=1.0,
        seed=42,
    )

    # Build trainer; fall back to tokenizer=tok if processing_class is unsupported
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": tr_ds,
        "eval_dataset": va_ds,
        "compute_metrics": compute_metrics,
    }
    try:
        trainer = Trainer(processing_class=tok, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tok, **trainer_kwargs)

    # Train and explicit evaluate (older versions don't auto-eval/save best)
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    # Manual save
    trainer.save_model(str(OUT_DIR))
    if hasattr(tok, "save_pretrained"):
        tok.save_pretrained(str(OUT_DIR))
    with open(OUT_DIR / "labels.txt", "w", encoding="utf-8") as f:
        for i, name in enumerate(le.classes_):
            f.write(f"{i}\t{name}\n")

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    # Save artifacts
    trainer.save_model(str(OUT_DIR))
    if hasattr(tok, "save_pretrained"):
        tok.save_pretrained(str(OUT_DIR))
    with open(OUT_DIR / "labels.txt", "w", encoding="utf-8") as f:
        for i, name in enumerate(le.classes_):
            f.write(f"{i}\t{name}\n")

if __name__ == "__main__":
    main()
