# src/train_optuna.py
import pandas as pd
import numpy as np
import optuna
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

# --- Convert SMILES to Morgan fingerprint ---
def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# --- Compute RDKit descriptors ---
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(Descriptors.descList))
    return np.array([f(mol) for _, f in Descriptors.descList], dtype=float)

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("./src/surfpro_imputed.csv").dropna(subset=["SMILES", "AW_ST_CMC"])

    # Convert SMILES → features
    fp_features = np.array([smiles_to_morgan(s) for s in df["SMILES"]])
    desc_features = np.array([smiles_to_descriptors(s) for s in df["SMILES"]])

    # Combine
    X = np.hstack([fp_features, desc_features])
    y = df["AW_ST_CMC"].values  # <-- change "Target" to your property column

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # --- Optuna Objective ---
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "n_jobs": -1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return -mean_squared_error(y_val, preds)  # minimize MSE

    # Run Optuna
    study = optuna.create_study(direction="maximize")  # maximizing -MSE = minimizing error
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("Best hyperparameters:", study.best_params)

    # --- Train final model with best params ---
    best_model = xgb.XGBRegressor(**study.best_params)
    best_model.fit(X_train, y_train)

    # Test evaluation
    y_pred_test = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\n📊 Final Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
