from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/raw/huggingface")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("🔹 Downloading HuggingFace dataset...")
dataset = load_dataset("dux-tecblic/symptom-disease-dataset")

df = dataset["train"].to_pandas()

out_path = OUT_DIR / "hf_symptom_disease.csv"
df.to_csv(out_path, index=False)

print(f"✅ Saved to {out_path}")
print("Columns:", df.columns.tolist())
print(df.head())
