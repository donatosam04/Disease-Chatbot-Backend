from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/raw/huggingface")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("🔹 Downloading mental health dataset (counsel-chat)...")

dataset = load_dataset("nbertagnolli/counsel-chat")

df = dataset["train"].to_pandas()

# Columns: questionText, topic
df = df.rename(columns={
    "questionText": "text",
    "topic": "label"
})

out_path = OUT_DIR / "mental_health_raw.csv"
df.to_csv(out_path, index=False)

print(f"✅ Saved → {out_path}")
print("Label distribution:")
print(df["label"].value_counts())
