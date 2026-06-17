import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM STYLE
# =========================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #1c1f2e 0%, #0f1117 60%);
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #6C63FF 0%, #4834d4 100%);
        padding: 2.2rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(108, 99, 255, 0.35);
        text-align: center;
    }
    .hero h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.4rem;
        margin: 0;
    }
    .hero p {
        color: #e6e6fa;
        margin-top: 0.4rem;
        font-size: 1.05rem;
    }

    /* Section card */
    .section-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(8px);
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #c7c2ff;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6C63FF 0%, #4834d4 100%);
        color: white;
        border-radius: 10px;
        padding: 0.7em 2.2em;
        border: none;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(108, 99, 255, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #7a72ff 0%, #5b46e8 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.55);
    }

    /* Result cards */
    .result-card {
        padding: 1.6rem;
        border-radius: 16px;
        margin-top: 1.2rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        animation: fadeIn 0.5s ease-in;
    }
    .high-risk {
        background: linear-gradient(135deg, #3a1a1a 0%, #4a1f1f 100%);
        color: #ff6b6b;
        border: 1px solid #ff6b6b;
        box-shadow: 0 4px 20px rgba(255, 107, 107, 0.25);
    }
    .low-risk {
        background: linear-gradient(135deg, #1a3a1f 0%, #1f4a28 100%);
        color: #6bff8e;
        border: 1px solid #6bff8e;
        box-shadow: 0 4px 20px rgba(107, 255, 142, 0.25);
    }
    .result-sub {
        font-size: 0.95rem;
        font-weight: 400;
        opacity: 0.85;
        margin-top: 0.4rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin-top: 1rem; }
    .metric-box {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-box .label { color: #9b9bb8; font-size: 0.85rem; }
    .metric-box .value { color: #ffffff; font-size: 1.3rem; font-weight: 700; margin-top: 0.2rem; }

    hr { border-color: rgba(255,255,255,0.08); }

    /* Fix label visibility for input widgets */
    label, .stSelectbox label, .stSlider label, label p {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Selectbox dropdown text */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.06) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }

    /* Slider value text */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #c7c2ff !important;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL & ENCODERS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
    columns = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))
    le_dict = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
    return model, columns, le_dict

model, columns, le_dict = load_artifacts()

# =========================
# HERO HEADER
# =========================
st.markdown("""
    <div class="hero">
        <h1>📊 Customer Churn Predictor</h1>
        <p>Get an instant, AI-powered estimate of customer churn risk</p>
    </div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR - QUICK INFO
# =========================
with st.sidebar:
    st.markdown("### ℹ️ How it works")
    st.write(
        "Fill in the customer's profile, account details, and billing "
        "information, then click **Predict Churn Risk** to see the "
        "model's prediction and probability score."
    )
    st.markdown("---")
    st.markdown("### 🎯 Tips")
    st.write("• Long-term, low-fee, high-engagement customers tend to churn less.")
    st.write("• Month-to-month contracts and electronic check payments raise risk.")
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

# =========================
# INPUT FORM
# =========================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Customer Profile</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
with c2:
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
with c3:
    partner = st.selectbox("Has Partner", ["No", "Yes"])
with c4:
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📡 Account & Services</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with c2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💳 Billing Details</div>', unsafe_allow_html=True)

payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

c1, c2 = st.columns(2)
with c1:
    monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 70)
with c2:
    total_charges = st.slider("Total Charges ($)", 0, 10000, 1000)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# BUILD INPUT ROW
# =========================
raw_input = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

input_df = pd.DataFrame([raw_input])

# =========================
# FEATURE ENGINEERING (same as training)
# =========================
input_df["AvgSpend"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)

service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]

input_df["ServicesCount"] = (
    input_df[service_cols]
    .replace({"No internet service": "No", "No phone service": "No"})
    .apply(lambda x: (x == "Yes").sum(), axis=1)
)

input_df["LongTermCustomer"] = (input_df["tenure"] > 24).astype(int)
input_df["PaymentRisk"] = (input_df["PaymentMethod"] == "Electronic check").astype(int)
input_df["SupportRisk"] = (input_df["TechSupport"] == "No").astype(int)
input_df["ContractRisk"] = input_df["Contract"].map({"Month-to-month": 2, "One year": 1, "Two year": 0})
input_df["ChargesPerService"] = input_df["MonthlyCharges"] / (input_df["ServicesCount"] + 1)

# =========================
# ENCODE CATEGORICAL COLUMNS using saved encoders
# =========================
for col, encoder in le_dict.items():
    if col in input_df.columns:
        val = input_df[col].iloc[0]
        classes = list(encoder.classes_)
        if val in classes:
            input_df[col] = encoder.transform([val])[0]
        else:
            # Try case-insensitive / stripped match before falling back
            match = next((c for c in classes if str(c).strip().lower() == str(val).strip().lower()), None)
            if match is not None:
                input_df[col] = encoder.transform([match])[0]
            else:
                input_df[col] = 0  # fallback for genuinely unseen category

# Ensure all columns are numeric (avoids silent dtype issues in predict_proba)
for col in input_df.columns:
    input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

# =========================
# ALIGN COLUMN ORDER
# =========================
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[columns]

# =========================
# DEBUG PANEL
# =========================
with st.expander("🔍 Debug: View model input & encoder details"):
    st.write("**Raw input (before encoding):**")
    st.json({k: (v.item() if hasattr(v, "item") else v) for k, v in raw_input.items()})

    st.write("**Final encoded row sent to model:**")
    st.dataframe(input_df)

    st.write("**Model expected columns:**")
    st.write(list(columns))

    st.write("**Encoder classes per column:**")
    for col, encoder in le_dict.items():
        st.write(f"- `{col}`:", list(encoder.classes_))

# =========================
# PREDICTION
# =========================
st.markdown("---")
center = st.columns([1, 1.4, 1])[1]

with center:
    predict_clicked = st.button("🔮 Predict Churn Risk", use_container_width=True)

if predict_clicked:
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.markdown(
            f'''<div class="result-card high-risk">
                ⚠️ High Churn Risk — {prob:.0%}
                <div class="result-sub">This customer is likely to churn. Consider proactive retention offers.</div>
            </div>''',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'''<div class="result-card low-risk">
                ✅ Low Churn Risk — {prob:.0%}
                <div class="result-sub">This customer appears loyal. Standard engagement should suffice.</div>
            </div>''',
            unsafe_allow_html=True
        )

    st.progress(float(prob))

    # Quick snapshot metrics
    st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="label">Tenure</div>
                <div class="value">{tenure} mo</div>
            </div>
            <div class="metric-box">
                <div class="label">Monthly Charges</div>
                <div class="value">${monthly_charges}</div>
            </div>
            <div class="metric-box">
                <div class="label">Contract</div>
                <div class="value">{contract}</div>
            </div>
            <div class="metric-box">
                <div class="label">Churn Probability</div>
                <div class="value">{prob:.1%}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)