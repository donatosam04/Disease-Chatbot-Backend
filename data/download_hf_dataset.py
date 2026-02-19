from datasets import load_dataset
import pandas as pd
from pathlib import Path

print("🔹 Loading HuggingFace dataset...")
dataset = load_dataset("dux-tecblic/symptom-disease-dataset")

# Convert to DataFrame
df = dataset["train"].to_pandas()

# Create directory
out_dir = Path("data/raw/huggingface")
out_dir.mkdir(parents=True, exist_ok=True)

# Save CSV
out_path = out_dir / "symptom_disease.csv"
df.to_csv(out_path, index=False)

print(f"✅ Saved to {out_path}")
print("Columns:", list(df.columns))
print(df.head())
