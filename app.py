"""
Housing Price Predictor — Streamlit UI

Prediction runs in a subprocess (predict_worker.py) so that TensorFlow, Keras,
CLIP, and XGBoost are completely isolated from Streamlit's server process.
All ML-library warnings and slow model loads happen in the child process;
the WebSocket that keeps the browser connected is never blocked or crashed.
"""

import json
import os
import subprocess
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()   # make RENTCAST_API_KEY available to the worker subprocess

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ── Session state — persists results across Streamlit reruns ──────────────────
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏠 Housing Price Predictor")
st.caption(
    "Structured model is the primary driver. "
    "Providing an address unlocks RentCast AVM. "
    "Uploading photos adds CLIP condition scoring."
)

# ── Property details ───────────────────────────────────────────────────────────
st.subheader("Property Details")

col1, col2, col3 = st.columns(3)
with col1:
    bed  = st.number_input("Bedrooms",       min_value=0, max_value=20,    value=3)
with col2:
    bath = st.number_input("Bathrooms",      min_value=0, max_value=20,    value=2)
with col3:
    sqft = st.number_input("Square Footage", min_value=100, max_value=30000, value=1500)

col4, col5 = st.columns(2)
with col4:
    city  = st.text_input("City", value="Los Angeles")
with col5:
    state = st.text_input(
        "State (full name)", value="California",
        help="Use full state name — e.g. 'California' not 'CA'",
    )

col6, col7 = st.columns(2)
with col6:
    address = st.text_input(
        "Street Address (optional)",
        placeholder="e.g. 123 Main St",
        help="Enables RentCast AVM cross-check",
    )
with col7:
    zip_code = st.text_input(
        "Zip Code (optional)",
        placeholder="e.g. 90001",
    )

# ── Image upload ───────────────────────────────────────────────────────────────
st.subheader("House Photos (optional, up to 5)")
st.caption("Upload exterior and/or interior photos. CLIP scores each for condition and averages the result.")

image_files = st.file_uploader(
    "Upload house images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# Enforce the 5-photo cap
if len(image_files) > 5:
    st.warning("Maximum 5 photos — only the first 5 will be used.")
    image_files = image_files[:5]

if image_files:
    cols = st.columns(min(len(image_files), 5))
    for col, f in zip(cols, image_files):
        with col:
            st.image(Image.open(f), caption=f.name, width="stretch")
            f.seek(0)

# ── Predict button ─────────────────────────────────────────────────────────────
if st.button("Estimate Price", type="primary", use_container_width=True):

    if not city or not state:
        st.error("City and State are required.")
        st.session_state.prediction = None
    else:
        input_dict = {
            "bed":   int(bed),
            "bath":  float(bath),
            "sqft":  float(sqft),
            "city":  city.strip(),
            "state": state.strip(),
        }
        if address.strip():
            input_dict["address"] = address.strip()
        if zip_code.strip():
            input_dict["zip_code"] = zip_code.strip()

        # Save each uploaded image to its own temp file so the worker can read them
        image_paths = []
        for f in image_files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.write(f.read())
            tmp.flush()
            tmp.close()
            image_paths.append(tmp.name)

        with st.spinner("Estimating price… (first run loads models, ~20 s)"):
            try:
                payload = json.dumps({"input_dict": input_dict, "image_paths": image_paths})

                # Run predict_worker in a child process.
                # capture_output=True means ALL ML warnings go to result.stderr
                # and never appear in Streamlit's terminal or crash its server.
                result = subprocess.run(
                    [sys.executable, "-m", "src.inference.predict_worker"],
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )

                if not result.stdout.strip():
                    error_detail = result.stderr[-800:] if result.stderr else "no output"
                    raise RuntimeError(f"Worker produced no output.\n{error_detail}")

                data = json.loads(result.stdout)

                if not data.get("ok"):
                    raise RuntimeError(data.get("error", "unknown error"))

                price     = data["price"]
                log_lines = data["log_lines"]

                sources = ["✅ Structured model"]
                sources.append("✅ RentCast AVM" if address.strip() else "⚪ AVM (no address)")
                n_photos = len(image_paths)
                sources.append(
                    f"✅ CLIP condition ({n_photos} photo{'s' if n_photos != 1 else ''})"
                    if image_paths else "⚪ Condition (no photos)"
                )

                st.session_state.prediction = {
                    "price":     price,
                    "log_lines": log_lines,
                    "sources":   sources,
                }

            except subprocess.TimeoutExpired:
                st.session_state.prediction = {"error": "Prediction timed out (>5 min)."}
            except Exception as exc:
                st.session_state.prediction = {"error": str(exc)}

# ── Results — rendered from session_state, survives widget interactions ────────
pred = st.session_state.prediction
if pred:
    st.divider()

    if "error" in pred:
        st.error(f"Prediction failed: {pred['error']}")

    else:
        price = pred["price"]

        # Price banner
        st.metric(label="Estimated Price", value=f"${price:,.0f}")

        # Signal badges
        bcols = st.columns(len(pred["sources"]))
        for col, src in zip(bcols, pred["sources"]):
            with col:
                if src.startswith("✅"):
                    st.success(src)
                else:
                    st.warning(src)

        # Prediction breakdown — plain text_area avoids StreamlitSyntaxHighlighter.js
        if pred.get("log_lines"):
            with st.expander("📊 Prediction breakdown", expanded=True):
                st.text_area(
                    label="breakdown",
                    value="\n".join(pred["log_lines"]),
                    height=340,
                    disabled=True,
                    label_visibility="collapsed",
                )
