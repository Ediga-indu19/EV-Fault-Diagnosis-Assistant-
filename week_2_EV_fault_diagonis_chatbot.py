import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Streamlit settings
st.set_page_config(page_title="EV Fault Diagnosis Chatbot", page_icon="⚙️", layout="centered")

st.title("⚙️ EV Fault Diagnosis Assistant — Final Version 🤖")
st.markdown("Predicts **EV fault types** using sensor data with confidence gauge, charts, and report generation.")

# -----------------------------------------------------
# File paths
# -----------------------------------------------------
DATA_PATH = "/Users/noorshaik/Downloads/EV_Battery_Fault_Diagnosis.csv"
MODEL_PATH = "/Users/noorshaik/Downloads/fault_chatbot_model.pkl"

# -----------------------------------------------------
# Train model
# -----------------------------------------------------
def train_model():
    df = pd.read_csv(DATA_PATH).dropna()
    features = ['Voltage (V)', 'Current (A)', 'Temperature (°C)', 'Motor Speed (RPM)',
                'Estimated SOC (%)', 'Ground Truth SOC (%)', 'Residual (%)']
    target = 'Fault Label'

    X = df[features]
    y = df[target]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump((model, le), MODEL_PATH)
    return model, le

# -----------------------------------------------------
# Load or train model
# -----------------------------------------------------
if os.path.exists(MODEL_PATH):
    data = joblib.load(MODEL_PATH)
    if isinstance(data, tuple):
        model, le = data
    else:
        model = data
        le = LabelEncoder()
    st.sidebar.success("✅ Model loaded successfully")
else:
    if os.path.exists(DATA_PATH):
        st.sidebar.info("Training model for the first time...")
        model, le = train_model()
        st.sidebar.success("✅ Model trained and saved successfully")
    else:
        st.sidebar.error("❌ Dataset not found! Please place the CSV file in the same folder.")
        st.stop()

# -----------------------------------------------------
# Sidebar controls
# -----------------------------------------------------
st.sidebar.header("⚙️ Options")
if st.sidebar.button("🔁 Retrain Model"):
    model, le = train_model()
    st.sidebar.success("✅ Model retrained successfully")

uploaded_file = st.sidebar.file_uploader("📂 Upload dataset (optional retrain)", type=["csv"])
if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    df_new.to_csv(DATA_PATH, index=False)
    st.sidebar.success("✅ New dataset uploaded and saved")

# -----------------------------------------------------
# Input fields
# -----------------------------------------------------
st.subheader("💬 Chat with your EV Assistant")

col1, col2 = st.columns(2)
with col1:
    voltage = st.number_input("Voltage (V)", 0.0, 10.0, 3.6)
    current = st.number_input("Current (A)", 0.0, 10.0, 3.5)
    temp = st.number_input("Temperature (°C)", 0.0, 100.0, 25.0)
    rpm = st.number_input("Motor Speed (RPM)", 0.0, 5000.0, 1300.0)
with col2:
    est_soc = st.number_input("Estimated SOC (%)", 0.0, 100.0, 90.0)
    gt_soc = st.number_input("Ground Truth SOC (%)", 0.0, 100.0, 89.0)
    residual = st.number_input("Residual (%)", 0.0, 10.0, 1.0)

input_data = pd.DataFrame([[voltage, current, temp, rpm, est_soc, gt_soc, residual]],
                          columns=['Voltage (V)', 'Current (A)', 'Temperature (°C)',
                                   'Motor Speed (RPM)', 'Estimated SOC (%)',
                                   'Ground Truth SOC (%)', 'Residual (%)'])

# -----------------------------------------------------
# Prediction + Confidence Gauge
# -----------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=input_data.columns.tolist() + ['Predicted Fault', 'Confidence'])

if st.button("🔍 Diagnose Fault"):
    pred_probs = model.predict_proba(input_data)
    pred_class = np.argmax(pred_probs)
    confidence = np.max(pred_probs) * 100
    label = le.inverse_transform([pred_class])[0]

    st.success(f"🧠 Predicted Fault Type: **{label}** ({confidence:.2f}% confident)")

    # Confidence Gauge (Progress circle)
    st.markdown("### 🔵 Confidence Level")
    st.progress(int(confidence))
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Dynamic advice
    if label.lower() == "normal":
        st.success("✅ System looks normal! Keep monitoring performance.")
    elif "warn" in label.lower():
        st.warning("⚠️ Warning detected! Check temperature and SOC difference.")
    elif "fault" in label.lower():
        st.error("❌ Fault detected! Inspect cooling and motor systems immediately.")
    else:
        st.info("ℹ️ Unknown label — please verify input values.")

    # Save to session history
    new_entry = input_data.copy()
    new_entry['Predicted Fault'] = label
    new_entry['Confidence'] = confidence
    st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)

    # SOC comparison bar chart
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(['Estimated SOC', 'Ground Truth SOC'], [est_soc, gt_soc], color=['blue', 'orange'])
    ax.set_title("SOC Comparison")
    st.pyplot(fig)

# -----------------------------------------------------
# Prediction history
# -----------------------------------------------------
st.subheader("📋 Prediction History")
if not st.session_state.history.empty:
    st.dataframe(st.session_state.history.tail(10))
    csv = st.session_state.history.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Prediction History", csv, "EV_Fault_Predictions.csv", "text/csv")

# -----------------------------------------------------
# Batch Prediction
# -----------------------------------------------------
st.subheader("🗂 Batch Prediction")
batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batchupload")
if batch_file is not None:
    df_batch = pd.read_csv(batch_file)
    required_cols = ['Voltage (V)', 'Current (A)', 'Temperature (°C)', 'Motor Speed (RPM)',
                     'Estimated SOC (%)', 'Ground Truth SOC (%)', 'Residual (%)']
    if all(c in df_batch.columns for c in required_cols):
        preds = model.predict(df_batch[required_cols])
        confs = np.max(model.predict_proba(df_batch[required_cols]), axis=1) * 100
        df_batch['Predicted Fault'] = le.inverse_transform(preds)
        df_batch['Confidence (%)'] = confs
        st.dataframe(df_batch.head(20))
        st.download_button("⬇️ Download Batch Results",
                           df_batch.to_csv(index=False).encode('utf-8'),
                           "Batch_Fault_Predictions.csv", "text/csv")
    else:
        st.error(f"CSV missing required columns: {required_cols}")

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.divider()
st.caption("Developed by Noor — Week-2 Internship Final Submission 💻")
