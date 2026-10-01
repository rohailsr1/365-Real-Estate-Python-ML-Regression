import datetime
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Theme (Navy & Gold)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Saleem Real Estate - Dubai Valuation Engine",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Styling with Balanced Font Sizes & Centered Logos
st.markdown("""
<style>
/* Global Font & Space Balance */
html, body, [data-testid="stAppViewContainer"] {
    font-size: 14px !important;
}

.stApp {
    background-color: #F4F6F9 !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 96% !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0A192F !important;
    border-right: 2px solid #C5A059 !important;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #C5A059 !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    margin-top: 0.3rem !important;
    margin-bottom: 0.3rem !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="input"] > div {
    background-color: #112240 !important;
    border: 1px solid #C5A059 !important;
    border-radius: 6px !important;
    color: #FFFFFF !important;
    min-height: 36px !important;
}

[data-testid="stSidebar"] input {
    color: #FFFFFF !important;
    font-size: 0.88rem !important;
}

/* Centered Main Header Container */
.brand-header-container {
    background-color: #FFFFFF;
    padding: 0.8rem 1.2rem;
    border-radius: 10px;
    border-bottom: 3px solid #C5A059;
    box-shadow: 0 3px 8px rgba(10, 25, 47, 0.06);
    margin-bottom: 0.8rem;
    text-align: center;
}

.brand-title {
    font-size: 1.8rem;
    font-weight: 900;
    color: #0A192F;
    letter-spacing: 1.2px;
    margin: 0;
    line-height: 1.15;
    font-family: 'Helvetica Neue', sans-serif;
}

.brand-subtitle {
    font-size: 0.85rem;
    font-weight: 700;
    color: #C5A059;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

.brand-tagline {
    font-size: 0.78rem;
    color: #64748B;
    font-style: italic;
    margin-top: 0.15rem;
}

/* Property Specification Badges */
.spec-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #C5A059;
    border-radius: 8px;
    padding: 0.55rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
}

.spec-card .spec-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.spec-card .spec-value {
    font-size: 1rem;
    font-weight: 800;
    color: #0A192F;
    margin-top: 0.15rem;
}

/* Primary Action Button */
.stButton>button {
    background: linear-gradient(135deg, #C5A059 0%, #9A7B31 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 0.55rem 1.5rem !important;
    border-radius: 8px !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 3px 8px rgba(197, 160, 89, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #0A192F 0%, #112240 100%) !important;
    color: #C5A059 !important;
    box-shadow: 0 5px 12px rgba(10, 25, 47, 0.25) !important;
}

/* Metric Display Cards */
h1, h2, h3, h4, .stMarkdown p {
    color: #0A192F !important;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-left: 5px solid #0A192F !important;
    padding: 0.65rem 1rem !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.03) !important;
}

div[data-testid="stMetricLabel"] p {
    color: #64748B !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
}

div[data-testid="stMetricValue"] div {
    color: #0A192F !important;
    font-weight: 800 !important;
    font-size: 1.35rem !important;
}

/* Range Breakdown Cards (Total & PSF) */
.val-box {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.65rem;
    text-align: center;
    box-shadow: 0 2px 5px rgba(0,0,0,0.03);
}

.val-box.baseline {
    background-color: #0A192F;
    border: 2px solid #C5A059;
    box-shadow: 0 4px 10px rgba(10, 25, 47, 0.15);
}

.val-box .title {
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #C5A059;
}

.val-box.baseline .title {
    color: #C5A059;
}

.val-box .amount {
    font-size: 1.2rem;
    font-weight: 900;
    color: #0A192F;
    margin-top: 0.2rem;
}

.val-box.baseline .amount {
    color: #FFFFFF;
}

.section-label {
    font-size: 0.85rem;
    font-weight: 800;
    color: #0A192F;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.6rem;
    margin-bottom: 0.35rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Model & Artifact Loader
# -----------------------------------------------------------------------------
MODEL_ARTIFACT_PATH = 'uae_real_estate_avm_tuned_model.joblib'
LOGO_PATH = 'logo.png'

@st.cache_resource
def load_valuation_artifacts():
    try:
        artifact = joblib.load(MODEL_ARTIFACT_PATH)
        return artifact
    except Exception as e:
        st.error(f"Error loading model artifact '{MODEL_ARTIFACT_PATH}': {e}")
        st.stop()

artifact = load_valuation_artifacts()
model = artifact['model']
encoder = artifact['encoder']
cat_cols = artifact['categorical_cols']
feature_names = artifact['feature_names']

# Categorical Option Lists
ROOM_MAPPING = {
    'Studio': 0, 'Single Room': 1, '1 B/R': 2, '2 B/R': 3, '2 B/R+M': 4,
    '3 B/R': 5, '3 B/R+M': 6, '4 B/R': 7, '4 B/R+M': 8, '5 B/R': 9,
    '5 B/R+M': 10, '6 B/R': 11, 'Penthouse': 12
}

COMMUNITIES = [
    'Jumeirah Village Circle (JVC)', 'Dubai Marina', 'Downtown Dubai',
    'Mohammed Bin Rashid City (MBR)', 'Business Bay', 'Palm Jumeirah',
    'Jumeirah Lakes Towers (JLT)', 'Dubai Hills Estate', 'Dubai Silicon Oasis (DSO)',
    'Dubai Production City (IMPZ)', 'Jumeirah Village Triangle (JVT)', 'Al Sufouh',
    'Al Furjan / Discovery Gardens', 'International City', 'Dubai Creek Harbour',
    'Al Barsha 1', 'Al Warqaa'
]

LANDMARKS = [
    'Burj Khalifa', 'Burj Al Arab', 'Dubai Frame', 'Global Village',
    'Expo City Dubai', 'Ain Dubai', 'Museum Of The Future', 'Dubai Opera',
    'Atlantis The Palm', 'Downtown Dubai', 'Dubai Water Canal', 'Unknown'
]

MALLS = [
    'Dubai Mall', 'Mall Of The Emirates', 'Dubai Marina Mall',
    'Nakheel Mall', 'Dubai Hills Mall', 'Ibn Battuta Mall',
    'City Centre Mirdif', 'Mercato Shopping Mall', 'Dubai Outlet Mall', 'Unknown'
]

BASE_MARKET_DATE = datetime.date(2020, 1, 1)

# -----------------------------------------------------------------------------
# 3. Sidebar Configuration (Centered Logo)
# -----------------------------------------------------------------------------
if os.path.exists(LOGO_PATH):
    sb_col1, sb_col2, sb_col3 = st.sidebar.columns([1, 2.5, 1])
    with sb_col2:
        st.image(LOGO_PATH, use_container_width=True)
else:
    st.sidebar.markdown("""
<div style="text-align: center; padding: 0.2rem 0;">
    <h2 style="color: #C5A059; margin-top: 0.2rem; font-size: 1.1rem;">SALEEM REAL ESTATE</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("📋 Specifications")

property_type = st.sidebar.selectbox("Property Archetype", ["Unit", "Villa"])

reg_type = st.sidebar.radio("Registration Status", ["Ready Properties", "Off-Plan Properties"])
is_off_plan = 1 if "Off-Plan" in reg_type else 0

rooms_label = st.sidebar.selectbox("Bedrooms / Layout", list(ROOM_MAPPING.keys()), index=3)
rooms_ordinal = ROOM_MAPPING[rooms_label]

size_sqft = st.sidebar.number_input(
    "Property Size (Sq. Ft.)",
    min_value=100.0,
    max_value=50000.0,
    value=1200.0,
    step=50.0
)

st.sidebar.header("📍 Location")

community = st.sidebar.selectbox("Market Community", COMMUNITIES, index=0)
nearest_landmark = st.sidebar.selectbox("Nearest Landmark", LANDMARKS, index=0)
nearest_mall = st.sidebar.selectbox("Nearest Mall", MALLS, index=0)

# -----------------------------------------------------------------------------
# 4. Main Page Branding & Summary Badges (Centered Logo)
# -----------------------------------------------------------------------------
if os.path.exists(LOGO_PATH):
    logo_col1, logo_col2, logo_col3 = st.columns([1.2, 1, 1.2])
    with logo_col2:
        st.image(LOGO_PATH, use_container_width=True)

st.markdown("""
<div class="brand-header-container">
    <div class="brand-title">SALEEM REAL ESTATE</div>
    <div class="brand-subtitle">Automated Real Estate Valuation Engine</div>
    <div class="brand-tagline">"Finding the right place for you"</div>
</div>
""", unsafe_allow_html=True)

# Specification Badges
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown(f"""
<div class="spec-card">
    <div class="spec-label">Selected Layout</div>
    <div class="spec-value">{rooms_label} ({property_type})</div>
</div>
""", unsafe_allow_html=True)

with col_info2:
    st.markdown(f"""
<div class="spec-card">
    <div class="spec-label">Total Area</div>
    <div class="spec-value">{size_sqft:,.0f} sq. ft.</div>
</div>
""", unsafe_allow_html=True)

with col_info3:
    st.markdown(f"""
<div class="spec-card">
    <div class="spec-label">Community</div>
    <div class="spec-value">{community}</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# 5. Prediction Execution & Range Output
# -----------------------------------------------------------------------------
if st.button("🚀 Calculate Property Valuation Range", use_container_width=True):
    # Dynamic current date parameters calculated behind the scenes for model input
    current_now = datetime.datetime.now()
    trans_year = current_now.year
    trans_month = current_now.month
    trans_quarter = (trans_month - 1) // 3 + 1
    market_age_months = (trans_year - BASE_MARKET_DATE.year) * 12 + (trans_month - BASE_MARKET_DATE.month)

    input_data = pd.DataFrame([{
        'log_size_sqft': np.log(size_sqft),
        'rooms_ordinal': rooms_ordinal,
        'is_off_plan': is_off_plan,
        'transaction_year': trans_year,
        'transaction_quarter': trans_quarter,
        'transaction_month': trans_month,
        'market_age_months': market_age_months,
        'property_type_en': property_type,
        'market_community_name': community,
        'nearest_landmark_en': nearest_landmark,
        'nearest_mall_en': nearest_mall
    }])

    try:
        # Encode features & maintain feature ordering
        input_encoded = encoder.transform(input_data)
        input_encoded = input_encoded[feature_names]

        # Log prediction -> AED
        log_price_pred = model.predict(input_encoded)[0]
        actual_price_pred = np.exp(log_price_pred)

        # Valuation Range (13.52% MAPE)
        mape_margin = 0.1352
        lower_bound = actual_price_pred * (1 - mape_margin)
        upper_bound = actual_price_pred * (1 + mape_margin)

        # Per Sq. Ft. Calculations
        psf_baseline = actual_price_pred / size_sqft
        psf_lower = lower_bound / size_sqft
        psf_upper = upper_bound / size_sqft

        # Top Metric Summary Cards
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric(
                label="ESTIMATED PRICE RANGE (AED)",
                value=f"AED {lower_bound:,.0f} – {upper_bound:,.0f}",
                delta=f"Midpoint: AED {actual_price_pred:,.0f}"
            )

        with res_col2:
            st.metric(
                label="ESTIMATED PRICE / SQ. FT. RANGE",
                value=f"AED {psf_lower:,.0f} – {psf_upper:,.0f}/sqft",
                delta=f"Midpoint: AED {psf_baseline:,.2f}"
            )

        # ---------------------------------------------------------------------
        # 1. Total Price Valuation 3-Box Breakdown
        # ---------------------------------------------------------------------
        st.markdown('<div class="section-label">Total Valuation Breakdown</div>', unsafe_allow_html=True)
        t_min_col, t_mid_col, t_max_col = st.columns(3)

        with t_min_col:
            st.markdown(f"""
<div class="val-box">
    <div class="title">Conservative Estimate</div>
    <div class="amount">AED {lower_bound:,.0f}</div>
</div>
""", unsafe_allow_html=True)

        with t_mid_col:
            st.markdown(f"""
<div class="val-box baseline">
    <div class="title">Baseline Valuation</div>
    <div class="amount">AED {actual_price_pred:,.0f}</div>
</div>
""", unsafe_allow_html=True)

        with t_max_col:
            st.markdown(f"""
<div class="val-box">
    <div class="title">Optimistic Estimate</div>
    <div class="amount">AED {upper_bound:,.0f}</div>
</div>
""", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 2. Per Square Feet Valuation 3-Box Breakdown
        # ---------------------------------------------------------------------
        st.markdown('<div class="section-label">Price / Sq. Ft. Breakdown</div>', unsafe_allow_html=True)
        psf_min_col, psf_mid_col, psf_max_col = st.columns(3)

        with psf_min_col:
            st.markdown(f"""
<div class="val-box">
    <div class="title">Conservative / Sq. Ft.</div>
    <div class="amount">AED {psf_lower:,.0f}/sqft</div>
</div>
""", unsafe_allow_html=True)

        with psf_mid_col:
            st.markdown(f"""
<div class="val-box baseline">
    <div class="title">Baseline / Sq. Ft.</div>
    <div class="amount">AED {psf_baseline:,.0f}/sqft</div>
</div>
""", unsafe_allow_html=True)

        with psf_max_col:
            st.markdown(f"""
<div class="val-box">
    <div class="title">Optimistic / Sq. Ft.</div>
    <div class="amount">AED {psf_upper:,.0f}/sqft</div>
</div>
""", unsafe_allow_html=True)

    except Exception as err:
        st.error(f"Error during prediction execution: {err}")