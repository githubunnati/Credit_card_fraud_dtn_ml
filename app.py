import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
import sys
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    .main-title {font-size:2.4rem; font-weight:800; color:#1a1a2e;}
    .sub-title  {font-size:1.1rem; color:#555; margin-bottom:1.5rem;}
    .metric-box {
        background:#f0f4ff; border-radius:12px; padding:1rem 1.5rem;
        border-left:5px solid #4f6df5; margin-bottom:1rem;
    }
    .fraud-badge    {background:#ffe0e0;color:#c0392b;padding:4px 12px;border-radius:20px;font-weight:700;}
    .legit-badge    {background:#d5f5e3;color:#1e8449;padding:4px 12px;border-radius:20px;font-weight:700;}
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "model"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    return model, scaler, features


def preprocess_uploaded(df: pd.DataFrame):
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]  = df["trans_date_trans_time"].dt.hour
    df["day"]   = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["dob"]   = pd.to_datetime(df["dob"])
    df["age"]   = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 +
        (df["long"] - df["merch_long"]) ** 2
    )
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ["category", "gender"]:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def make_single_input(amt, category_idx, gender_idx, city_pop,
                      hour, day, month, age, distance, merch_lat, merch_long):
    return pd.DataFrame([[
        amt, category_idx, gender_idx, city_pop,
        hour, day, month, age, distance, merch_lat, merch_long
    ]], columns=[
        "amt","category","gender","city_pop",
        "hour","day","month","age","distance","merch_lat","merch_long"
    ])


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=80)
    st.title("🛡️ Fraud Detector")
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("Go to", ["🏠 Home", "🔍 Single Transaction", "📂 Batch Prediction", "📈 Model Info"])
    st.markdown("---")

    # Train button
    if not os.path.exists(os.path.join(MODEL_DIR, "fraud_model.pkl")):
        st.warning("⚠️ Model not trained yet!")
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training… this may take a few minutes."):
                result = subprocess.run(
                    [sys.executable, "fraud_model.py"],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("Model trained & saved!")
                st.cache_resource.clear()
            else:
                st.error(f"Error:\n{result.stderr[-500:]}")
    else:
        st.success("✅ Model Ready")
        if st.button("🔄 Re-train Model"):
            with st.spinner("Re-training…"):
                result = subprocess.run(
                    [sys.executable, "fraud_model.py"],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("Model updated!")
                st.cache_resource.clear()
            else:
                st.error(result.stderr[-500:])

# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<p class="main-title">💳 Credit Card Fraud Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">ML-powered system to classify transactions as Fraudulent or Legitimate</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.info("🔍 **Single Transaction**\nCheck one transaction manually")
    c2.info("📂 **Batch Prediction**\nUpload a CSV and get results")
    c3.info("📈 **Model Info**\nSee performance metrics")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. **Train** the model using the sidebar button (first time only).
2. Go to **Single Transaction** to test one transaction.
3. Go to **Batch Prediction** to upload your CSV and download results.
    """)

# ─────────────────────────────────────────────
# SINGLE TRANSACTION
# ─────────────────────────────────────────────
elif page == "🔍 Single Transaction":
    st.markdown("## 🔍 Single Transaction Check")
    st.markdown("Fill in the transaction details below:")

    categories = ["food_dining","gas_transport","grocery_net","grocery_pos",
                  "health_fitness","home","kids_pets","misc_net","misc_pos",
                  "personal_care","shopping_net","shopping_pos","travel","entertainment"]

    c1, c2, c3 = st.columns(3)
    with c1:
        amt       = st.number_input("Transaction Amount (₹/$)", min_value=0.01, value=150.0, step=0.01)
        category  = st.selectbox("Category", categories)
        gender    = st.selectbox("Gender", ["M", "F"])
    with c2:
        city_pop  = st.number_input("City Population", min_value=100, value=50000, step=100)
        hour      = st.slider("Transaction Hour (0-23)", 0, 23, 12)
        day       = st.slider("Day of Month", 1, 31, 15)
        month     = st.slider("Month", 1, 12, 6)
    with c3:
        age       = st.number_input("Cardholder Age", min_value=18, max_value=100, value=35)
        distance  = st.number_input("Distance (cardholder ↔ merchant)", min_value=0.0, value=2.5, step=0.1)
        merch_lat = st.number_input("Merchant Latitude",  value=33.99)
        merch_long= st.number_input("Merchant Longitude", value=-81.20)

    if st.button("🔎 Predict", use_container_width=True):
        if not os.path.exists(os.path.join(MODEL_DIR, "fraud_model.pkl")):
            st.error("Please train the model first from the sidebar!")
        else:
            model, scaler, features = load_model()
            cat_idx = categories.index(category)
            gen_idx = 0 if gender == "F" else 1
            X = make_single_input(amt, cat_idx, gen_idx, city_pop,
                                  hour, day, month, age, distance, merch_lat, merch_long)
            X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
            pred  = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0][1]

            st.markdown("---")
            if pred == 1:
                st.error(f"🚨 **FRAUDULENT TRANSACTION** — Fraud Probability: `{proba*100:.1f}%`")
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION** — Fraud Probability: `{proba*100:.1f}%`")

            st.progress(float(proba))

# ─────────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────────
elif page == "📂 Batch Prediction":
    st.markdown("## 📂 Batch Prediction")
    st.markdown("Upload a CSV file with the same columns as the training data (`fraudTest.csv`).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded, index_col=0)
        st.write(f"**Rows:** {len(df_raw):,}")
        st.dataframe(df_raw.head(5))

        if st.button("⚡ Run Predictions", use_container_width=True):
            if not os.path.exists(os.path.join(MODEL_DIR, "fraud_model.pkl")):
                st.error("Please train the model first from the sidebar!")
            else:
                model, scaler, features = load_model()
                with st.spinner("Preprocessing & predicting…"):
                    df_proc = preprocess_uploaded(df_raw)
                    X = df_proc[features]
                    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
                    preds  = model.predict(X_scaled)
                    probas = model.predict_proba(X_scaled)[:, 1]

                df_raw["predicted_fraud"] = preds
                df_raw["fraud_probability"] = (probas * 100).round(2)

                fraud_count = preds.sum()
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", f"{len(df_raw):,}")
                col2.metric("🚨 Fraudulent", f"{fraud_count:,}")
                col3.metric("✅ Legitimate", f"{len(df_raw)-fraud_count:,}")

                st.dataframe(
                    df_raw[["trans_date_trans_time","amt","merchant","category",
                             "predicted_fraud","fraud_probability"]].head(50)
                )

                csv_out = df_raw.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv_out,
                    "fraud_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )

                if "is_fraud" in df_raw.columns:
                    st.markdown("---")
                    st.markdown("### 📊 Evaluation (since labels exist in data)")
                    from sklearn.metrics import classification_report
                    report = classification_report(
                        df_raw["is_fraud"], preds,
                        target_names=["Legitimate","Fraud"], output_dict=True
                    )
                    st.dataframe(pd.DataFrame(report).transpose())

# ─────────────────────────────────────────────
# MODEL INFO
# ─────────────────────────────────────────────
elif page == "📈 Model Info":
    st.markdown("## 📈 Model Information")
    if not os.path.exists(os.path.join(MODEL_DIR, "fraud_model.pkl")):
        st.warning("Model not trained yet. Train it from the sidebar first.")
    else:
        model, scaler, features = load_model()
        st.markdown(f"**Model type:** `{type(model).__name__}`")
        st.markdown(f"**Features used ({len(features)}):**")
        st.code(", ".join(features))

        st.markdown("---")
        st.markdown("### 🔑 Feature Importance (if applicable)")
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            st.bar_chart(fi)
        elif hasattr(model, "coef_"):
            fi = pd.Series(np.abs(model.coef_[0]), index=features).sort_values(ascending=False)
            st.bar_chart(fi)
        else:
            st.info("Feature importance not available for this model type.")

        st.markdown("---")
        st.markdown("### Algorithms Compared")
        st.markdown("""
| Algorithm | Strength | Best For |
|---|---|---|
| Logistic Regression | Fast, interpretable | Baseline |
| Decision Tree | Easy to visualize | Small datasets |
| **Random Forest** | High accuracy, handles imbalance | **Fraud detection ✅** |
        """)
