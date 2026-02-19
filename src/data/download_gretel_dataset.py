from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/raw/huggingface")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("🔹 Downloading GretelAI dataset...")
dataset = load_dataset("gretelai/symptom_to_diagnosis")

df = dataset["train"].to_pandas()

# Normalize column names
df = df.rename(columns={
    "symptoms": "text",
    "diagnosis": "label"
})

out_path = OUT_DIR / "gretel_symptom_to_diagnosis.csv"
df.to_csv(out_path, index=False)

print(f"✅ Saved to {out_path}")
print("Columns:", df.columns.tolist())
print(df.head())
