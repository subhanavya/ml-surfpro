# Perry's Handbook Style ML Prediction Project

This project fetches chemical property data from PubChem and NIST WebBook, merges it into a Perry's Handbook-style dataset, 
and trains multiple machine learning models to predict physical properties.

## Steps
1. Prepare a list of chemical names or CAS numbers in `data/chemicals.txt`.
2. Run `src/fetch_pubchem.py` to get PubChem data.
3. Run `src/fetch_nist.py` to get NIST data.
4. Run `src/build_dataset.py` to merge into a CSV.
5. Run `src/train.py` to train and compare ML models.
