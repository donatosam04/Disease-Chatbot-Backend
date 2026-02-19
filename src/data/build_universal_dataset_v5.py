import pandas as pd
import json
from pathlib import Path

# ============================
# CONFIG
# ============================

BASE_DATASET = "data/processed/universal_disease_train_v4.csv"
MENDELEY = "data/processed/mendeley_cleaned.csv"
GRETEL = "data/processed/gretel_normalized.csv"

LABEL_MAP_FILE = "src/data/label_normalization_map.json"

OUT_PATH = "data/processed/universal_disease_train_v5.csv"

MAX_SAMPLES_PER_DISEASE = 800  # safe cap

Path("data/processed").mkdir(parents=True, exist_ok=True)

# ============================
# LOAD DATASETS
# ============================

print("🔹 Loading datasets...")

base_df = pd.read_csv(BASE_DATASET)[["text", "label"]]
mendeley_df = pd.read_csv(MENDELEY)[["text", "label"]]
gretel_df = pd.read_csv(GRETEL)[["text", "label"]]

# ============================
# COMBINE
# ============================

combined = pd.concat(
    [base_df, mendeley_df, gretel_df],
    ignore_index=True
)

# ============================
# CLEAN TEXT & LABELS
# ============================

combined["text"] = (
    combined["text"]
    .astype(str)
    .str.lower()
    .str.replace("_", " ", regex=False)
    .str.strip()
)

combined["label"] = (
    combined["label"]
    .astype(str)
    .str.lower()
    .str.strip()
)

# ============================
# NORMALIZE LABELS (CRITICAL FIX)
# ============================

with open(LABEL_MAP_FILE, "r") as f:
    label_map = json.load(f)

combined["label"] = combined["label"].apply(
    lambda x: label_map.get(x, x)
)

print("🧼 Labels after normalization:", combined["label"].nunique())

# ============================
# BALANCE DATASET
# ============================

print("🔹 Balancing dataset...")

balanced = (
    combined
    .groupby("label", group_keys=False)
    .apply(
        lambda x: x.sample(
            n=min(len(x), MAX_SAMPLES_PER_DISEASE),
            random_state=42
        )
    )
    .reset_index(drop=True)
)

# ============================
# STATS
# ============================

print("\n✅ Total samples:", len(balanced))
print("✅ Unique diseases:", balanced["label"].nunique())
print("\nLabel distribution:")
print(balanced["label"].value_counts())

# ============================
# SAVE
# ============================

balanced.to_csv(OUT_PATH, index=False)
print(f"\n💾 Saved → {OUT_PATH}")
