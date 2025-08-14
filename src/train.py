# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

if __name__ == "__main__":
    df = pd.read_csv("data/perry_dataset.csv").dropna()

    # Example: Predict BoilingPoint_K from other features
    X = df.drop(columns=["Name", "BoilingPoint_K"])
    y = df["BoilingPoint_K"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "SVM": SVR(),
        "LinearRegression": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05)
    }

    best_model = None
    best_score = -float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = r2_score(y_val, preds)
        print(f"{name} R² on validation: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = name

    print(f"\n🏆 Best model: {best_model} with R²={best_score:.4f}")
