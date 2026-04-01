import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb


# ======================================================
# CONFIG
# ======================================================

DATA_PATH = "data/processed/dataset_c_cleaned.csv"
MODEL_SAVE_PATH = "outputs/models/structured_xgb.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


# ======================================================
# LOAD DATA
# ======================================================

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ======================================================
# TRAIN / VAL / TEST SPLIT (BEFORE ENCODING)
# ======================================================

train_df, temp_df = train_test_split(
    df,
    test_size=TEST_SIZE + VAL_SIZE,
    random_state=RANDOM_STATE
)

val_ratio_adjusted = VAL_SIZE / (TEST_SIZE + VAL_SIZE)

val_df, test_df = train_test_split(
    temp_df,
    test_size=1 - val_ratio_adjusted,
    random_state=RANDOM_STATE
)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))


# ======================================================
# TARGET ENCODING (TRAIN SET ONLY)
# ======================================================

print("Applying target encoding...")

city_means = train_df.groupby("city")["log_price"].mean()

def apply_target_encoding(df, city_map):
    df = df.copy()
    df["city_encoded"] = df["city"].map(city_map)
    return df

train_df = apply_target_encoding(train_df, city_means)
val_df = apply_target_encoding(val_df, city_means)
test_df = apply_target_encoding(test_df, city_means)

# Handle unseen cities
global_mean = train_df["log_price"].mean()

train_df["city_encoded"].fillna(global_mean, inplace=True)
val_df["city_encoded"].fillna(global_mean, inplace=True)
test_df["city_encoded"].fillna(global_mean, inplace=True)


# ======================================================
# FEATURE SELECTION
# ======================================================

FEATURES = ["bed", "bath", "sqft", "city_encoded"]

X_train = train_df[FEATURES]
X_val = val_df[FEATURES]
X_test = test_df[FEATURES]

y_train = train_df["log_price"]
y_val = val_df["log_price"]
y_test = test_df["log_price"]


# ======================================================
# SCALING
# ======================================================

print("Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ======================================================
# MODEL INITIALIZATION
# ======================================================

print("Building XGBoost model...")

model = xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE
)


# ======================================================
# TRAINING
# ======================================================

print("Training model...")

model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=True
)


# ======================================================
# EVALUATION
# ======================================================

def evaluate(model, X, y, name="Dataset"):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    print(f"\n{name} Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    return rmse, r2


print("\nEvaluating model...")

evaluate(model, X_train_scaled, y_train, "Train")
evaluate(model, X_val_scaled, y_val, "Validation")
evaluate(model, X_test_scaled, y_test, "Test")


# ======================================================
# SAVE MODEL ARTIFACTS
# ======================================================

print("Saving model artifacts...")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

joblib.dump({
    "model": model,
    "scaler": scaler,
    "city_encoding_map": city_means,
    "global_mean": global_mean,
    "features": FEATURES
}, MODEL_SAVE_PATH)

print("Structured model saved successfully.")