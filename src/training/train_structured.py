import pandas as pd
import json
import os
import joblib
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.structured_model import (
    prepare_data,
    split_data,
    train_models,
    evaluate_models,
    select_best_model
)


# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = "data/processed/structured_b_clean.csv"
MODEL_PATH = "outputs/models"
LOG_PATH = "outputs/logs"


def ensure_dirs():
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)


def main():
    ensure_dirs()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Preparing data...")
    X, y = prepare_data(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training models...")
    models = train_models(X_train, y_train)

    print("Evaluating models...")
    results = evaluate_models(models, X_test, y_test)

    print("\nModel Results:")
    print(json.dumps(results, indent=2))

    best_name, best_model = select_best_model(models, results)
    print(f"\nBest Model: {best_name}")

    print("Saving model...")
    joblib.dump(best_model, f"{MODEL_PATH}/structured_model.pkl")

    print("Saving metrics...")
    with open(f"{LOG_PATH}/structured_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()