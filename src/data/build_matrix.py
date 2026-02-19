import pandas as pd
from pathlib import Path
import json

RAW_DIR = Path("data/raw/kaggle_disease_symptoms")
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MATRIX = OUT_DIR / "disease_symptom_matrix.parquet"
OUT_SYMPTOMS = OUT_DIR / "symptom_list.json"

def load_raw():
    files = list(RAW_DIR.glob("*.csv"))
    assert files, f"No CSV files found in {RAW_DIR}"
    # Pick the largest CSV as the main table
    main = max(files, key=lambda p: p.stat().st_size)
    df = pd.read_csv(main)
    return df

def unify(df: pd.DataFrame):
    # Choose disease column
    disease_col = "Disease" if "Disease" in df.columns else df.columns[0]
    # Symptom columns are all non-disease columns
    sym_cols = [c for c in df.columns if c != disease_col]

    # Normalize disease names
    df[disease_col] = df[disease_col].astype(str).str.strip().str.lower()

    # Convert symptom columns to binary 0/1
    for c in sym_cols:
        s = df[c]
        # handle 0/1, Y/N, text, NaN
        if s.dtype == object:
            sc = s.fillna("").astype(str).str.lower()
            binv = sc.isin(["1", "y", "yes", "true"]) | sc.str.contains("present|yes|true|1")
            df[c] = binv.astype(int)
        else:
            df[c] = (s.fillna(0).astype(float) > 0).astype(int)

    # Deduplicate by disease (union symptoms)
    dfu = df.groupby(disease_col)[sym_cols].max().reset_index()
    return dfu, disease_col, sym_cols

def main():
    df = load_raw()
    matrix, disease_col, sym_cols = unify(df)
    matrix.to_parquet(OUT_MATRIX, index=False)
    with OUT_SYMPTOMS.open("w", encoding="utf-8") as f:
        json.dump(sym_cols, f, ensure_ascii=False, indent=2)
    print(f"Saved matrix: {OUT_MATRIX}")
    print(f"Saved symptoms: {OUT_SYMPTOMS}")
    print(f"Diseases: {matrix.shape[0]} | Symptoms: {len(sym_cols)}")

if __name__ == "__main__":
    main()
