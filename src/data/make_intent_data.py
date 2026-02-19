import pandas as pd
import numpy as np
from pathlib import Path
import json, random

MATRIX = Path("data/interim/disease_symptom_matrix.parquet")
SYMPTOMS = Path("data/interim/symptom_list.json")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

OUT_TRAIN = PROC_DIR / "bert_intent_train.csv"
OUT_VAL   = PROC_DIR / "bert_intent_val.csv"
OUT_VEC   = PROC_DIR / "ml_symptom_vectors.parquet"

def to_user_text(sym_cols, row, max_sym=6):
    present = [s.replace("_"," ").replace("-"," ") for s in sym_cols if int(row[s]) == 1]
    random.shuffle(present)
    present = present[:max_sym] if present else ["no clear symptoms"]
    return "I have " + ", ".join(present)

def main():
    df = pd.read_parquet(MATRIX)
    with open(SYMPTOMS, "r", encoding="utf-8") as f:
        sym_cols = json.load(f)

    disease_col = df.columns[0]  # first column is disease after build step
    records = []
    # cap examples per disease for quick training/demo
    MAX_PER_DISEASE = 3
    for dis, grp in df.groupby(disease_col):
        sample = grp.head(MAX_PER_DISEASE)
        for _, r in sample.iterrows():
            text = to_user_text(sym_cols, r)
            records.append({"text": text, "label": dis})

    data = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    m = int(0.8 * len(data))
    data.iloc[:m].to_csv(OUT_TRAIN, index=False)
    data.iloc[m:].to_csv(OUT_VAL, index=False)

    # emit full matrix for later ML layers
    ml = df.rename(columns={disease_col: "disease"})
    ml.to_parquet(OUT_VEC, index=False)

    print(f"Wrote: {OUT_TRAIN} ({m} rows), {OUT_VAL} ({len(data)-m} rows)")
    print(f"Wrote: {OUT_VEC} (shape {ml.shape})")

if __name__ == "__main__":
    main()
