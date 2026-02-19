import pandas as pd, json, random
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

MATRIX = Path("data/interim/disease_symptom_matrix.parquet")
SYMPTOMS = Path("data/interim/symptom_list.json")
OUT_TRAIN = Path("data/processed/bert_intent_train_reduced.csv")
OUT_VAL = Path("data/processed/bert_intent_val_reduced.csv")

TOP_K = 100
PER_DISEASE = 40
MAX_SYM = 8

PREFIX = ["I have", "I am experiencing", "I feel", "Symptoms include"]
SEVERITY = ["mild", "moderate", "severe"]
DURATION = ["since yesterday", "for two days", "for a week", "for several hours", "with sudden onset"]

def to_text(sym_cols, row, max_sym=MAX_SYM):
    present = [s.replace("_"," ").replace("-"," ") for s in sym_cols if int(row[s])==1]
    if not present:
        present = ["no clear symptoms"]
    random.shuffle(present)
    present = present[:max_sym]
    pref = random.choice(PREFIX)
    if random.random() < 0.6:
        present[0] = f"{random.choice(SEVERITY)} {present[0]}"
    if random.random() < 0.6:
        present.append(random.choice(DURATION))
    return f"{pref} " + ", ".join(present)

def main():
    df = pd.read_parquet(MATRIX)
    sym_cols = json.loads(Path(SYMPTOMS).read_text(encoding="utf-8"))
    df["sym_count"] = df[sym_cols].sum(axis=1)
    topk = df.sort_values("sym_count", ascending=False).head(TOP_K).drop(columns=["sym_count"])
    disease_col = topk.columns[0]

    rows = []
    for _, r in topk.iterrows():
        for _ in range(PER_DISEASE):
            rows.append({"text": to_text(sym_cols, r), "label": r[disease_col]})
    data = pd.DataFrame(rows)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(data["text"], data["label"]))
    train = data.iloc[train_idx].reset_index(drop=True)
    val = data.iloc[val_idx].reset_index(drop=True)

    train.to_csv(OUT_TRAIN, index=False)
    val.to_csv(OUT_VAL, index=False)
    print(f"Wrote: {OUT_TRAIN} ({len(train)}), {OUT_VAL} ({len(val)}), classes={topk.shape[0]}")

if __name__ == "__main__":
    main()
