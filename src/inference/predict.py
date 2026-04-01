import numpy as np
import pandas as pd
import joblib
import torch
from src.models.cnn_model import load_model, get_transform, extract_features


# -----------------------------
# LOAD MODELS (once)
# -----------------------------
structured_model = joblib.load("outputs/models/structured_model.pkl")
adjustment_model = joblib.load("outputs/models/image_adjustment_model.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model = load_model(device)
transform = get_transform()


# -----------------------------
# STRUCTURED PREDICTION
# -----------------------------
def predict_structured(input_dict):
    df = pd.DataFrame([input_dict])

    X = df.copy()

    # Encode city/state exactly like training
    X = pd.get_dummies(X, columns=["city", "state"], drop_first=True)

    # Align with training features
    model_features = structured_model.feature_names_in_
    X = X.reindex(columns=model_features, fill_value=0)

    pred = structured_model.predict(X)[0]

    return pred


# -----------------------------
# IMAGE ADJUSTMENT
# -----------------------------
def predict_adjustment(image_path, input_dict, structured_pred):
    # Extract CNN features
    features = extract_features(cnn_model, image_path, transform, device)

    feature_df = pd.DataFrame([features])

    # Add structured features
    feature_df["bed"] = input_dict["bed"]
    feature_df["bath"] = input_dict["bath"]
    feature_df["sqft"] = input_dict["sqft"]
    feature_df["city"] = input_dict["city"]

    # 🚨 ADD THIS (CRITICAL)
    feature_df["structured_pred"] = structured_pred

    # Encode city
    feature_df = pd.get_dummies(feature_df, columns=["city"], drop_first=True)

    # Align columns
    feature_df = feature_df.reindex(
        columns=adjustment_model.feature_names_in_,
        fill_value=0
    )

    adjustment = adjustment_model.predict(feature_df)[0]

    print("Expected features:", len(adjustment_model.feature_names_in_))
    print("Actual features:", feature_df.shape[1])

    return adjustment


# -----------------------------
# FINAL PREDICTION
# -----------------------------
def predict_price(input_dict, image_path=None):
    
    # Step 1: base prediction
    structured_pred = predict_structured(input_dict)

    # Step 2: optional image adjustment
    if image_path:
        adjustment = predict_adjustment(image_path, input_dict, structured_pred)
    else:
        adjustment = 0

    # Step 3: combine
    final_log_price = structured_pred + adjustment
    final_price = np.expm1(final_log_price)

    print("Structured (log):", structured_pred)
    print("Adjustment:", adjustment)
    print("Final log:", structured_pred + adjustment)

    return final_price