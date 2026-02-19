import pandas as pd
from pathlib import Path

# ============================
# LOAD DATA
# ============================

hf = pd.read_csv("data/raw/huggingface/symptom_disease.csv")
label_map = pd.read_csv("data/mappings/hf_selected_labels.csv")

# ============================
# CLEAN LABEL MAP SAFELY
# ============================

# Convert hf_label to numeric, invalid → NaN
label_map["hf_label"] = pd.to_numeric(
    label_map["hf_label"],
    errors="coerce"
)

# Drop invalid rows
label_map = label_map.dropna(subset=["hf_label"])

# Convert to int
label_map["hf_label"] = label_map["hf_label"].astype(int)

# ============================
# ENSURE HF LABEL TYPE
# ============================

hf["label"] = hf["label"].astype(int)

# ============================
# MERGE
# ============================

hf_clean = hf.merge(
    label_map,
    left_on="label",
    right_on="hf_label",
    how="inner"
)

# ============================
# FINAL CLEANUP
# ============================

hf_clean = hf_clean[["text", "disease_name"]]
hf_clean = hf_clean.rename(columns={"disease_name": "label"})
hf_clean = hf_clean.groupby("label").filter(lambda x: len(x) >= 20)

print("✅ Filtered samples:", len(hf_clean))
print("✅ Unique diseases:", hf_clean["label"].nunique())
print("\nLabel distribution:")
print(hf_clean["label"].value_counts())

# ============================
# SAVE
# ============================

out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "hf_cleaned.csv"
hf_clean.to_csv(out_path, index=False)

print(f"\n💾 Saved → {out_path}")
