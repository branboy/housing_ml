import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.fusion_model import build_fusion_model, evaluate_model


def train_fusion_pipeline():

    # Load data
    df = pd.read_csv("data/processed/dataset_c_cleaned.csv")
    condition_scores = np.load("data/processed/condition_scores.npy")

    # Features
    X_structured = df.drop(columns=["price", "log_price", "image_id", "street", "city"]).values
    y = df["log_price"].values

    X = np.hstack([X_structured, condition_scores])

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # Model
    model = build_fusion_model()
    model.fit(X_train, y_train)

    # Evaluate
    print("\nFusion Model Results:\n")
    evaluate_model(model, X_train, y_train, "Train")
    evaluate_model(model, X_val, y_val, "Validation")
    evaluate_model(model, X_test, y_test, "Test")

    # Save artifacts
    os.makedirs("outputs/models", exist_ok=True)

    joblib.dump(model, "outputs/models/fusion_model.pkl")
    joblib.dump(scaler, "outputs/models/scaler.pkl")

    print("\nModel and scaler saved successfully.")


if __name__ == "__main__":
    train_fusion_pipeline()