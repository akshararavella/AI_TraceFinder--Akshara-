import streamlit as st
from PIL import Image
import numpy as np
import joblib
import io

# Load model and label encoder
model = joblib.load("LGBM_model_final.joblib")
le = joblib.load("LGBM_label_encoder_final.joblib")

st.title("🕵️‍♂️ AI TraceFinder: Image Forensics")
st.markdown("Upload an image to detect its source or authenticity.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to numpy array
    img_np = np.array(image)

    # Compute FFT features
    fft = np.fft.fft2(img_np)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    fft_mean = np.mean(magnitude)
    fft_std = np.std(magnitude)

    # Prepare feature vector
    features = np.array([[fft_mean, fft_std]])

    # Predict probabilities
    probs = model.predict_proba(features)[0]
    max_prob = np.max(probs)
    prediction_idx = np.argmax(probs)
    prediction_label = le.inverse_transform([prediction_idx])[0]

    # Map prediction to display label with confidence threshold
    scanner_labels = [label for label in le.classes_ if label != 'Tampered']
    if prediction_label == 'Tampered' and max_prob >= 0.6:
        display_label = "Tampered"
    elif prediction_label in scanner_labels and max_prob >= 0.6:
        display_label = "Flatfield Scanner"
    else:
        display_label = "Unknown"

    # Display result
    st.markdown("## 🔍 Analysis Result")
    st.markdown(f"### 🧠 Predicted Class: **{display_label}**")
    st.markdown(f"**Confidence:** {max_prob * 100:.2f}%")
