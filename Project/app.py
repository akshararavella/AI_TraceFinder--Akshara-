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
    st.image(image, caption="Uploaded Image", use_column_width=True)

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

    # Predict
    prediction_idx = model.predict(features)[0]
    prediction_label = le.inverse_transform([prediction_idx])[0]

    # Fallback logic: if prediction is not 'Tampered' and not a known scanner, label as 'Authentic'
    known_scanners = [label for label in le.classes_ if label != 'Tampered']
    if prediction_label not in known_scanners and prediction_label != 'Tampered':
        prediction_label = 'Authentic'

    # Display result
    st.markdown(f"### 🧠 Prediction: **{prediction_label}**")
