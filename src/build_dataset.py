# src/build_dataset.py
import pandas as pd

if __name__ == "__main__":
    pubchem_df = pd.read_csv("data/pubchem_data.csv")
    nist_df = pd.read_csv("data/nist_data.csv")

    merged = pd.merge(pubchem_df, nist_df, on="Name", how="outer")
    merged.to_csv("data/perry_dataset.csv", index=False)
    print("✅ Final Perry-style dataset saved to data/perry_dataset.csv")