import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def build_dataset(n_rows: int = 1200) -> pd.DataFrame:
    current_year = 2026

    # Year: cars from 2003 to 2025
    year = np.random.randint(2003, 2026, size=n_rows)

    # Present (showroom) price in lakhs, roughly 3 to 30 lakhs
    present_price = np.round(np.random.uniform(3, 30, size=n_rows), 2)

    # Kms driven: older cars tend to have more kms, with noise
    car_age = current_year - year
    kms_driven = (car_age * np.random.uniform(4000, 9000, size=n_rows)
                  + np.random.normal(0, 5000, size=n_rows))
    kms_driven = np.clip(kms_driven, 500, 250000).astype(int)

    
    depreciation = present_price * (0.91 ** car_age)
    kms_penalty = (kms_driven / 100000) * (present_price * 0.08)
    noise = np.random.normal(0, 0.4, size=n_rows)

    selling_price = depreciation - kms_penalty + noise
    selling_price = np.clip(selling_price, present_price * 0.10, present_price)
    selling_price = np.round(selling_price, 2)

    df = pd.DataFrame({
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Selling_Price": selling_price,
    })
    return df


def main():
    print("Building dataset...")
    df = build_dataset()
    df.to_csv("car_data.csv", index=False)
    print(f"Saved dataset -> car_data.csv  (shape={df.shape})")
    print(df.head(), "\n")

    feature_cols = ["Year", "Present_Price", "Kms_Driven"]
    target_col = "Selling_Price"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\nEvaluation on held-out test set:")
    print(f"  MAE : {mae:.3f} lakhs")
    print(f"  R^2 : {r2:.3f}")

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "current_year": 2026,
    }
    joblib.dump(artifact, "model.pkl")
    print("\nSaved trained model -> model.pkl")


if __name__ == "__main__":
    main()
