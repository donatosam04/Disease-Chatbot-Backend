import pandas as pd

df = pd.read_csv("data/raw/huggingface/symptom_disease.csv")

print("Total samples:", len(df))
print("Unique labels:", df["label"].nunique())
print("\nSample rows:")
print(df.head(10))

print("\nTop 10 most frequent labels:")
print(df["label"].value_counts().head(10))
