# Multi-Signal Housing Price Prediction

**XGBoost · RentCast AVM · CLIP Condition Scoring**

Brantson Bui & Hanru Li — Machine Learning (ML 4824) · April 2026

---

## Project Webpage

The project webpage is built into the Streamlit app. Once the app is running, click **Project Info** in the sidebar to access the full project page — including paper and slides downloads, pipeline overview, results, and team info.

**Live deployment:** `https://YOUR_APP_NAME.streamlit.app` *(update once deployed)*

---

## Running the App

### 1. Prerequisites

- Python 3.9 or higher
- A RentCast API key (free tier at [rentcast.io](https://rentcast.io)) — needed for AVM enrichment. The app works without it but AVM will be disabled.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU (optional but recommended):** If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`. The CLIP scorer and XGBoost training will run significantly faster.

### 3. Set your API key

Create a file named `.env` in the project root with the following line:

```
RENTCAST_API_KEY=your_key_here
```

If you skip this step the app will still run — AVM-based enrichment will simply be unavailable.

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501` in your browser.

---

## Folder Structure

```
housing_ml/
├── app.py                      # Streamlit application entry point
├── requirements.txt
├── index.html                  # Project webpage (open in browser)
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   ├── images/             # 15k CA training images
│   │   ├── realty_images/      # 89 multi-state CLIP calibration images
│   │   └── structured/         # Raw CSV data
│   └── processed/
│       ├── fusion_dataset.csv
│       ├── realty_manifest.csv
│       └── structured_clean.csv
├── outputs/
│   └── models/                 # Trained model files (.pkl, .json)
├── src/
│   ├── models/
│   │   ├── clip_condition.py   # CLIP zero-shot condition scorer
│   │   ├── fusion_model.py
│   │   └── structured_model.py
│   ├── training/
│   │   ├── calibrate_clip.py   # Re-run CLIP calibration
│   │   ├── train_structured.py
│   │   └── train_fusion.py
│   ├── inference/
│   └── utils/
└── notebooks/
    ├── eda.ipynb
    └── test.ipynb
```

---

## Re-training (optional)

The `outputs/models/` folder already contains all trained model files — you do not need to retrain to run the app. If you want to retrain from scratch:

```bash
# 1. Train the structured XGBoost model
python -m src.training.train_structured

# 2. Re-run CLIP calibration (requires images in data/raw/realty_images/)
python -m src.training.calibrate_clip

# 3. Train the fusion model
python -m src.training.train_fusion
```

---

## Deliverables

| File | Description |
|------|-------------|
| `app.py` | Streamlit app entry point (prediction tool) |
| `pages/1_Project_Info.py` | Project webpage — accessible from the app sidebar |
| `ieee_report.docx` | Final project report (IEEE double-column format) |
| `Project_presentation.pptx` | Presentation slides |
| `presentation_script.md` | Speaker notes for the presentation |

---

## Notes

- Do **not** rename or move the `data/` or `outputs/` folders — the app uses relative paths.
- The first run will download ~600 MB of CLIP model weights and cache them locally. Subsequent runs use the cache.
- The RentCast free tier allows 50 API calls/month. Results are cached in `outputs/rentcast_cache.json` so repeated lookups for the same address do not count against your quota.
