import streamlit as st
from src.inference.predict import predict_price

from PIL import Image
import tempfile

st.title("🏠 Housing Price Predictor")

st.write("Enter house details:")

# -----------------------------
# USER INPUTS
# -----------------------------
bed = st.number_input("Bedrooms", min_value=0, value=3)
bath = st.number_input("Bathrooms", min_value=0, value=2)
sqft = st.number_input("Square Footage", min_value=0, value=1500)

city = st.text_input("City", "Los Angeles")
state = st.text_input("State", "CA")

image_file = st.file_uploader("Upload house image (optional)", type=["jpg", "png"])


# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Price"):
    input_dict = {
        "bed": bed,
        "bath": bath,
        "sqft": sqft,
        "city": city,
        "state": state
    }

    image_path = None

    if image_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(image_file.read())
        image_path = temp_file.name

        st.image(Image.open(image_path), caption="Uploaded Image")

    price = predict_price(input_dict, image_path)

    st.success(f"💰 Estimated Price: ${price:,.2f}")