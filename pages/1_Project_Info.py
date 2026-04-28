"""
Project Info page — shown as "Project Info" in the Streamlit sidebar.
Serves as the public-facing project webpage: overview, pipeline, method,
results, team, and direct download buttons for paper, slides, and code.
"""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Project Info — Housing Price Prediction",
    page_icon="🏠",
    layout="wide",
)

# ── Helper: load file bytes for download buttons ──────────────────────────────
def load_file(path: str):
    p = Path(path)
    return p.read_bytes() if p.exists() else None

paper_bytes  = load_file("misc/Project_Report.pdf")
slides_bytes = load_file("misc/Project_presentation.pptx")
zip_bytes    = load_file("housing_ml.zip")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Hero */
  .hero {
    background: linear-gradient(135deg, #1a2744 0%, #1e3a5f 60%, #0f3460 100%);
    color: white;
    padding: 2.5rem 2rem 2rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1.5rem;
  }
  .hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 0.3rem; color: white; }
  .hero .sub { color: #93c5fd; font-size: 1rem; margin-bottom: 0.4rem; }
  .hero .authors { color: #cbd5e1; font-size: 0.95rem; }
  .hero .course  { color: #64748b; font-size: 0.82rem; margin-top: 0.2rem; }

  /* Stat cards */
  .stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.5rem; }
  .stat-card {
    flex: 1; min-width: 120px;
    background: #1a2744; border-radius: 10px;
    padding: 1rem; text-align: center; color: white;
  }
  .stat-card .val { font-size: 1.7rem; font-weight: 900; color: #93c5fd; }
  .stat-card .lbl { font-size: 0.72rem; color: #94a3b8; margin-top: 2px;
                    text-transform: uppercase; letter-spacing: .04em; }

  /* Section label */
  .sec-lbl {
    font-size: 0.72rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: #2563eb; margin-bottom: 4px;
  }

  /* Abstract */
  .abstract {
    background: #dbeafe;
    border-left: 4px solid #2563eb;
    border-radius: 0 10px 10px 0;
    padding: 1.1rem 1.4rem;
    font-size: 0.93rem; color: #1e3a5f; line-height: 1.7;
  }

  /* Pipeline steps */
  .pipeline { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 1rem 0; }
  .pipe-step {
    flex: 1; min-width: 130px;
    border: 1.5px solid #e2e8f0; border-radius: 10px;
    padding: 0.9rem 0.7rem; text-align: center; background: white;
  }
  .pipe-step .icon { font-size: 1.6rem; }
  .pipe-step h4 { font-size: 0.82rem; font-weight: 700; color: #1a2744; margin: 4px 0 2px; }
  .pipe-step p  { font-size: 0.72rem; color: #94a3b8; margin: 0; }
  .pipe-arrow   { font-size: 1.2rem; color: #94a3b8; padding: 0 2px; }
  .pipe-final   { background: #1a2744 !important; }
  .pipe-final h4{ color: white !important; }
  .pipe-final p { color: #93c5fd !important; }

  /* Result callout */
  .result-hero {
    background: linear-gradient(135deg, #0f3460, #1a2744);
    border-radius: 12px; padding: 1.5rem 1.8rem; color: white; margin-top: 1rem;
  }
  .result-hero .big { font-size: 2rem; font-weight: 900; color: #4ade80; }
  .result-hero .label { font-size: 0.78rem; color: #94a3b8; }

  /* Team card */
  .team-card {
    border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.4rem; text-align: center; background: white;
  }
  .avatar {
    width: 60px; height: 60px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; font-weight: 800; color: white;
    margin: 0 auto 0.7rem;
  }
  .av-blue   { background: linear-gradient(135deg, #2563eb, #0d9488); }
  .av-green  { background: linear-gradient(135deg, #0d9488, #16a34a); }

  /* Citation */
  .cite {
    background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'Courier New', monospace; font-size: 0.8rem;
    color: #475569; white-space: pre-wrap;
  }

  /* Hide default Streamlit padding a bit */
  .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="display:inline-block;background:rgba(255,255,255,0.12);
       border:1px solid rgba(255,255,255,0.2);border-radius:999px;
       padding:0.25rem 0.9rem;font-size:0.75rem;letter-spacing:.06em;
       text-transform:uppercase;color:#93c5fd;margin-bottom:0.8rem">
    Machine Learning · ML 4824 · April 2026
  </div>
  <h1>Multi-Signal Housing Price Prediction</h1>
  <div class="sub">XGBoost &nbsp;·&nbsp; RentCast AVM &nbsp;·&nbsp; CLIP Condition Scoring</div>
  <div class="authors">Brantson Bui &nbsp;&amp;&nbsp; Hanru Li</div>
  <div class="course">Machine Learning (ML 4824) &nbsp;·&nbsp; April 2026</div>
</div>
""", unsafe_allow_html=True)

# ── Download buttons ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    if zip_bytes:
        st.download_button("⬇️ Code (.zip)", zip_bytes,
                           file_name="housing_ml.zip",
                           mime="application/zip",
                           use_container_width=True)
    else:
        st.link_button("💻 Code", "https://github.com/branboy/housing_ml",
                       use_container_width=True)

with col2:
    if paper_bytes:
        st.download_button("📄 Paper (.pdf)", paper_bytes,
                           file_name="Project_Report.pdf",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           use_container_width=True)
    else:
        st.warning("project_report.docx not found")

with col3:
    if slides_bytes:
        st.download_button("📊 Slides (.pptx)", slides_bytes,
                           file_name="Project_presentation.pptx",
                           mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                           use_container_width=True)
    else:
        st.warning("Project_presentation.pptx not found")

with col4:
    st.page_link("app.py", label="🏠 Launch App", use_container_width=True)

st.divider()

# ── Stats ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
  <div class="stat-card"><div class="val">2.2M</div><div class="lbl">Training Homes</div></div>
  <div class="stat-card"><div class="val">54</div><div class="lbl">US States</div></div>
  <div class="stat-card"><div class="val">40+</div><div class="lbl">Features</div></div>
  <div class="stat-card"><div class="val">28.3%</div><div class="lbl">Median MAPE</div></div>
  <div class="stat-card"><div class="val">0.15%</div><div class="lbl">Best-Case Error</div></div>
  <div class="stat-card"><div class="val">3</div><div class="lbl">Signals Fused</div></div>
</div>
""", unsafe_allow_html=True)

# ── Abstract ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">Overview</div>', unsafe_allow_html=True)
st.markdown("### Abstract")
st.markdown("""
<div class="abstract">
  This project presents a multi-signal housing price prediction system that stacks three
  complementary signals: (1) an XGBoost structured model trained on 2.2 million US homes
  across 54 states, (2) a RentCast Automated Valuation Model (AVM) cross-checked with
  geography-aware blending weights, and (3) a CLIP zero-shot visual condition scorer that
  interprets uploaded property photos. The structured model achieves a <strong>28.3% median
  absolute percentage error (MAPE)</strong> on a held-out test split of 8,445 homes.
  An outlier detection mechanism identifies when the structured model undershoots premium
  markets and applies a graduated premium boost of up to +30%. When tested on a known sold
  home ($931,000 in Olney, MD), the full pipeline predicted $929,592 —
  <strong>0.15% error</strong>. The system is deployed as a Streamlit web application
  supporting structured inputs, optional street address for AVM enrichment, and photo
  uploads for condition scoring.
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Pipeline ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">Architecture</div>', unsafe_allow_html=True)
st.markdown("### Prediction Pipeline")
st.markdown("""
<div class="pipeline">
  <div class="pipe-step" style="border-color:#93c5fd">
    <div class="icon">🏗️</div><h4>Structured Model</h4>
    <p>XGBoost · 40+ features · Always active</p>
  </div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step" style="border-color:#6ee7b7">
    <div class="icon">📡</div><h4>RentCast AVM</h4>
    <p>Address-based · 10–60% blend</p>
  </div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step" style="border-color:#fca5a5">
    <div class="icon">📍</div><h4>Market Anchor</h4>
    <p>Zip price/sqft × sqft</p>
  </div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step" style="border-color:#c4b5fd">
    <div class="icon">🖼️</div><h4>CLIP Condition</h4>
    <p>ViT-B/32 · Zero-shot · ±15%</p>
  </div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step pipe-final">
    <div class="icon">💰</div><h4>Final Estimate</h4>
    <p>Blended · Explained</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Method ────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">Methodology</div>', unsafe_allow_html=True)
st.markdown("### How It Works")

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("**🏗️ Signal 1 — XGBoost Structured Model**")
    st.markdown("""
Trained on **2.2M homes** across 54 states on log(1 + price).

- **Geographic** — city/zip target encoding, state ordinal
- **Structural** — sqft, beds, baths, ratios, sqft², flags
- **Property** — age, decade bucket, type (5 categories)
- **Extras** — lot size, HOA, pool, garage, stories

`max_depth=5` · `min_child_weight=15` · CUDA-accelerated
""")

with m2:
    st.markdown("**📡 Signal 2 — RentCast AVM Blend**")
    st.markdown("""
When a street address is provided, RentCast returns an AVM estimate with confidence level.

- Low confidence → 10% blend weight
- High confidence (non-CA) → 60% blend weight
- Extra weight when AVM diverges >0.30 log-units
- Zip market anchor corroborates the AVM signal
""")

with m3:
    st.markdown("**🖼️ Signal 3 — CLIP Visual Condition**")
    st.markdown("""
CLIP ViT-B/32 scores uploaded photos **zero-shot** — no labeled training data required.

- 4 positive prompts (modern, renovated, high-end)
- 4 negative prompts (dated, worn, deferred maintenance)
- score = mean(pos) − mean(neg)
- Capped at ±15% · up to 5 photos averaged
""")

st.divider()

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">Evaluation</div>', unsafe_allow_html=True)
st.markdown("### Results")

r1, r2 = st.columns([1, 1])

with r1:
    st.markdown("**MAPE by Price Tier** — 8,445 held-out homes")
    st.dataframe(
        {
            "Price Tier":  ["$0–300k", "$300–600k", "$600k–1M", "$1M+"],
            "MAPE":        ["69.2%", "28.8% ✅", "31.0%", "39.2%"],
            "Median APE":  ["36.8%", "21.1%", "24.6%", "35.4%"],
            "n":           ["2,148", "2,145", "2,105", "2,047"],
        },
        hide_index=True,
        use_container_width=True,
    )

with r2:
    st.markdown("**Best & Worst States**")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("✅ Strongest")
        st.dataframe(
            {"State": ["Arizona", "Kansas", "Illinois", "Maryland", "Connecticut"],
             "MAPE":  ["28.8%", "31.1%", "31.4%", "32.6%", "33.4%"]},
            hide_index=True, use_container_width=True,
        )
    with c2:
        st.caption("⚠️ Hardest")
        st.dataframe(
            {"State": ["California", "Hawaii", "Florida", "DC", "Maine"],
             "MAPE":  ["54.5%", "54.5%", "51.8%", "46.4%", "45.4%"]},
            hide_index=True, use_container_width=True,
        )

# Full pipeline callout
st.markdown("""
<div class="result-hero">
  <div style="font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;
              color:#93c5fd;margin-bottom:.4rem">Full-Pipeline Validation</div>
  <div style="font-size:1.1rem;font-weight:700;color:white;margin-bottom:.8rem">
    Olney, MD — 4bd / 3ba / 2,552 sqft
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:2rem;align-items:center">
    <div><div class="big">$929,592</div><div class="label">Predicted</div></div>
    <div style="color:#64748b;font-size:1.5rem">vs.</div>
    <div><div style="font-size:2rem;font-weight:900;color:#cbd5e1">$931,000</div>
         <div class="label">Actual Sale Price</div></div>
    <div style="margin-left:auto">
      <div style="font-size:2.5rem;font-weight:900;color:#4ade80">0.15%</div>
      <div class="label">Prediction Error</div>
    </div>
  </div>
  <div style="color:#64748b;margin-top:.8rem;font-size:.82rem">
    Structured-only baseline: ~$370,000 (−60% error) · Outlier detection fired ·
    +23% premium boost · CLIP adjusted +2% from 3 photos
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Screenshots ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">Application</div>', unsafe_allow_html=True)
st.markdown("### App Screenshots")

from PIL import Image
imgs = [
    ("Maryland.png",   "Maryland — full pipeline: AVM + market + CLIP active"),
    ("Cupertino.png",  "Cupertino, CA — high-cost market with geographic encoding"),
    ("Texas.png",      "Texas — outlier detection + premium boost applied"),
    ("Blacksburg.png", "Blacksburg, VA — mid-market, model's strongest price tier"),
]
cols = st.columns(2)
for i, (fname, caption) in enumerate(imgs):
    p = Path(fname)
    if p.exists():
        with cols[i % 2]:
            st.image(str(p), caption=caption, use_container_width=True)

st.divider()

# ── Team ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">Authors</div>', unsafe_allow_html=True)
st.markdown("### Team")

t1, t2, _ = st.columns([1, 1, 1])
with t1:
    st.markdown("""
<div class="team-card">
  <div class="avatar av-blue">BB</div>
  <strong>Brantson Bui</strong><br>
  <span style="font-size:.78rem;color:#94a3b8">ML 4824 · April 2026</span>
  <ul style="text-align:left;font-size:.82rem;margin-top:.7rem;color:#475569">
    <li>XGBoost model &amp; feature engineering</li>
    <li>Multi-signal blending framework</li>
    <li>Outlier detection &amp; premium boost</li>
    <li>CLIP integration &amp; calibration</li>
    <li>Streamlit app development</li>
  </ul>
</div>
""", unsafe_allow_html=True)

with t2:
    st.markdown("""
<div class="team-card">
  <div class="avatar av-green">HL</div>
  <strong>Hanru Li</strong><br>
  <span style="font-size:.78rem;color:#94a3b8">ML 4824 · April 2026</span>
  <ul style="text-align:left;font-size:.82rem;margin-top:.7rem;color:#475569">
    <li>Data collection &amp; preprocessing</li>
    <li>Kaggle + Zillow + Realty API merge</li>
    <li>Stratified batch evaluation</li>
    <li>Per-tier &amp; per-state analysis</li>
    <li>Figures &amp; report writing</li>
  </ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Citation ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-lbl">Reference</div>', unsafe_allow_html=True)
st.markdown("### Citation")
st.markdown("""
<div class="cite">@misc{bui2026housing,
  title   = {Multi-Signal Housing Price Prediction:
             XGBoost, RentCast AVM, and CLIP Condition Scoring},
  author  = {Bui, Brantson and Li, Hanru},
  year    = {2026},
  note    = {ML 4824 Course Project, April 2026}
}</div>
""", unsafe_allow_html=True)
