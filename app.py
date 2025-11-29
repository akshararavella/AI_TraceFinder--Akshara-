import streamlit as st
from PIL import Image
import numpy as np
import joblib
import pandas as pd

# Load model and label encoder
model = joblib.load("LGBM_model_final.joblib")
le = joblib.load("LGBM_label_encoder_final.joblib")

# App title and header
st.set_page_config(page_title="🔍 TraceFinder", page_icon="🔍", layout="wide")
st.markdown(
    """
    <div style='text-align:center; padding-top:16px;'>
        <h1>🔍 TraceFinder</h1>
        <h4 style='color:#6a7ff7;'>Scanner Identification & Tamper Detection 🖨️</h4>
        <p style='color:#aaaaff; font-size:20px;'>Upload a scanned image below 👇</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("📁 Upload scanned image (PNG/JPG/JPEG/TIF/TIFF)", type=["png", "jpg", "jpeg", "tif", "tiff"])

def extract_fft_features(image):
    img_np = np.array(image)
    fft = np.fft.fft2(img_np)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    fft_mean = np.mean(magnitude)
    fft_std = np.std(magnitude)
    return np.array([[fft_mean, fft_std]])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Extract features and predict
    features = extract_fft_features(image)
    probs = model.predict_proba(features)[0]
    max_prob = np.max(probs)
    prediction_idx = np.argmax(probs)
    prediction_label = le.inverse_transform([prediction_idx])[0]

    # Map prediction to display label
    scanner_labels = [label for label in le.classes_ if label != 'Tampered']
    if prediction_label == 'Tampered' and max_prob >= 0.6:
        display_label = "Tampered"
    elif prediction_label in scanner_labels and max_prob >= 0.6:
        display_label = "Flatfield Scanner"
    else:
        display_label = "Authentic"

    # Display results with new background color
    st.markdown("## 🔍 Analysis Result")
    st.markdown(
        f"""
        <div style='padding:24px;border-radius:12px;
             background:#f0f4ff;border:2px solid #6a7ff7;'>
            <div style='font-size:22px;color:#3446c5;'>🖨️ Scanner Prediction</div>
            <div style='font-size:32px;margin-top:10px;font-weight:bold;color:#1c1c1c;'>{display_label}</div>
            <div style='font-size:16px;color:#5c5c5c;margin-top:14px;'>
                🎯 Confidence: <b>{max_prob * 100:.2f}%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Prepare CSV result
    result_df = pd.DataFrame({
        "File": [uploaded_file.name],
        "Predicted Class": [display_label],
        "Confidence (%)": [round(max_prob * 100, 2)]
    })

    st.download_button(
        label="📥 Download Results as CSV",
        data=result_df.to_csv(index=False),
        file_name="tracefinder_results.csv",
        mime="text/csv"
    )
else:
    st.info("💡 Drag-and-drop or browse to upload and analyze.")
