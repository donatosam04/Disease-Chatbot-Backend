import pandas as pd
from pathlib import Path

BASE_FILE = "data/processed/universal_disease_train_v3.csv"
GRETEL_FILE = "data/processed/gretel_normalized.csv"
OUT_FILE = "data/processed/universal_disease_train_v4.csv"

df_base = pd.read_csv(BASE_FILE)
df_gretel = pd.read_csv(GRETEL_FILE)

assert {"text", "label"}.issubset(df_base.columns)
assert {"text", "label"}.issubset(df_gretel.columns)

df_all = pd.concat([df_base, df_gretel], ignore_index=True)

print("✅ Total samples:", len(df_all))
print("✅ Unique diseases:", df_all["label"].nunique())
print("\nLabel distribution:")
print(df_all["label"].value_counts())

df_all.to_csv(OUT_FILE, index=False)
print(f"\n💾 Saved → {OUT_FILE}")
