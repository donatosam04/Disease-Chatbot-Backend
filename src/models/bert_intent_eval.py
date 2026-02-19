import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    top_k_accuracy_score,
    precision_recall_fscore_support,
)

# ============================================================
# 🔁 CONFIG
# ============================================================

MODEL_DIR = Path("models_saved/intent_gi5")

EVAL_MODE = "hard"   # "easy" or "hard"

VAL_CSV = {
    "easy": "data/processed/gi5_val.csv",
    "hard": "data/processed/gi5_hard_realistic.csv",
}[EVAL_MODE]

OUT_DIR = MODEL_DIR / "eval" / EVAL_MODE
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"🔍 Evaluation mode: {EVAL_MODE}")
print(f"📂 Outputs → {OUT_DIR}")

# ============================================================
# Helper plots
# ============================================================

def save_headline_metrics_table(top1, top3, macro_prf, micro_prf, out_png, out_csv):
    macro_p, macro_r, macro_f1, _ = macro_prf
    micro_p, micro_r, micro_f1, _ = micro_prf

    rows = [
        ["Top-1 accuracy", f"{top1*100:.2f}%"],
        ["Top-3 accuracy", f"{top3*100:.2f}%"],
        ["Macro precision", f"{macro_p*100:.2f}%"],
        ["Macro recall", f"{macro_r*100:.2f}%"],
        ["Macro F1", f"{macro_f1*100:.2f}%"],
        ["Micro precision", f"{micro_p*100:.2f}%"],
        ["Micro recall", f"{micro_r*100:.2f}%"],
        ["Micro F1", f"{micro_f1*100:.2f}%"],
    ]

    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close()


def save_headline_metrics_bar(top1, top3, macro_prf, micro_prf, out_png):
    macro_p, macro_r, macro_f1, _ = macro_prf
    micro_p, micro_r, micro_f1, _ = micro_prf

    names = ["Top-1", "Top-3", "Macro P", "Macro R", "Macro F1",
             "Micro P", "Micro R", "Micro F1"]
    vals = [top1, top3, macro_p, macro_r, macro_f1,
            micro_p, micro_r, micro_f1]

    plt.figure(figsize=(8.5, 4))
    bars = plt.bar(names, [v * 100 for v in vals])
    plt.ylim(0, 100)
    plt.title("Headline metrics")
    plt.ylabel("Percent")

    for b, v in zip(bars, vals):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1,
            f"{v*100:.1f}%",
            ha="center",
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

# ============================================================
# MAIN
# ============================================================

def main():

    va = pd.read_csv(VAL_CSV)
    assert {"text", "label"}.issubset(va.columns)

    print(f"🧪 Validation samples: {len(va)}")
    print(va["label"].value_counts())

    # Load label mapping
    id2label = {}
    with open(MODEL_DIR / "labels.txt") as f:
        for line in f:
            i, l = line.strip().split("\t")
            id2label[int(i)] = l

    classes = [id2label[i] for i in range(len(id2label))]
    n_classes = len(classes)

    va = va[va["label"].isin(classes)].reset_index(drop=True)
    le = LabelEncoder().fit(classes)
    y_true = le.transform(va["label"])

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device).eval()

    probs, preds = [], []

    for t in tqdm(va["text"], desc="Predict"):
        enc = tok(
            t,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            logits = mdl(**enc).logits

        p = F.softmax(logits, dim=-1).cpu().numpy()[0]
        probs.append(p)
        preds.append(p.argmax())

    probs = np.vstack(probs)
    preds = np.array(preds)

    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=220)
    plt.close()

    # Metrics
    macro = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
    )
    micro = precision_recall_fscore_support(
        y_true, preds, average="micro", zero_division=0
    )

    top1 = top_k_accuracy_score(y_true, probs, k=1)
    top3 = top_k_accuracy_score(
        y_true,
        probs,
        k=min(3, n_classes)
    )

    save_headline_metrics_table(
        top1, top3, macro, micro,
        OUT_DIR / "headline_metrics_table.png",
        OUT_DIR / "headline_metrics_table.csv"
    )

    save_headline_metrics_bar(
        top1, top3, macro, micro,
        OUT_DIR / "headline_metrics_bar.png"
    )

    print("✅ Evaluation complete")

if __name__ == "__main__":
    main()
