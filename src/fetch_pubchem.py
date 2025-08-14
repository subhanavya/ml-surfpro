# Placeholder script - to be filled with actual implementation
# src/fetch_pubchem.py
import pubchempy as pcp
import pandas as pd
import time

def fetch_pubchem_properties(chemicals):
    results = []
    for chem in chemicals:
        try:
            compound = pcp.get_compounds(chem, 'name')[0]
            props = {
                "Name": chem,
                "MolecularWeight_gmol": compound.molecular_weight,
                "BoilingPoint_K": compound.boiling_point if hasattr(compound, 'boiling_point') else None,
                "MeltingPoint_K": compound.melting_point if hasattr(compound, 'melting_point') else None,
                "Density_kgm3": compound.density if hasattr(compound, 'density') else None
            }
            results.append(props)
        except Exception as e:
            print(f"Error fetching {chem}: {e}")
        time.sleep(0.2)  # avoid rate limits
    return pd.DataFrame(results)

if __name__ == "__main__":
    with open("data/chemicals.txt") as f:
        chemicals = [line.strip() for line in f if line.strip()]

    df_pubchem = fetch_pubchem_properties(chemicals)
    df_pubchem.to_csv("data/pubchem_data.csv", index=False)
    print("✅ PubChem data saved to data/pubchem_data.csv")


