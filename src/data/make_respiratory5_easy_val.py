import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA = "data/processed/respiratory5_bert.csv"
OUT_VAL = "data/processed/respiratory5_val.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

# Stratified split to preserve class balance
_, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

val_df.to_csv(OUT_VAL, index=False)

print("✅ Easy validation CSV created:")
print(val_df["label"].value_counts())
print(f"Saved → {OUT_VAL}")
