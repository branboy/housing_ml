import streamlit as st
import pandas as pd
from PIL import Image

from src.inference.predict import predict_price


# Load dataset columns (IMPORTANT)
df = pd.read_csv("data/processed/dataset_c_cleaned.csv")
feature_columns = df.drop(columns=["price", "log_price", "image_id", "street", "city"]).columns


st.title("🏠 Housing Price Predictor (AI)")

st.write("Enter property details and upload an image.")

use_images = st.checkbox("Use image in prediction", value=True)

# --- User Inputs ---
bed = st.number_input("Bedrooms", min_value=0, value=3)
bath = st.number_input("Bathrooms", min_value=0, value=2)
sqft = st.number_input("Square Footage", min_value=0, value=1500)

city = st.selectbox("City", options=[col for col in feature_columns if "city_" in col])


# Build structured input dict
user_input = {
    "bed": bed,
    "bath": bath,
    "sqft": sqft,
    city: 1   # one-hot encoding
}

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload house image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Price"):
        if use_images:
            if image is None:
                st.warning("Please upload an image.")
            else:
                price = predict_price(user_input, image, feature_columns)
                st.success(f"Estimated Price (with image): ${price:,.2f}")
        else:
            import joblib
            import numpy as np

            structured_model = joblib.load("outputs/models/structured_xgb.pkl")
            scaler = joblib.load("outputs/models/scaler.pkl")

            x = np.zeros(len(feature_columns))

            for i, col in enumerate(feature_columns):
                if col in user_input:
                    x[i] = user_input[col]

            x = scaler.transform([x])

            log_price = structured_model.predict(x)[0]
            price = np.exp(log_price)

            st.success(f"Estimated Price (structured only): ${price:,.2f}")