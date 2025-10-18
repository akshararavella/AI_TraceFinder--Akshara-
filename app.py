import streamlit as st
from PIL import Image
import numpy as np
import joblib
import pandas as pd

# Load model and label encoder from Project/models/
model = joblib.load("Project/models/LGBM_model_final.joblib")
le = joblib.load("Project/models/LGBM_label_encoder_final.joblib")

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

    # Threshold-based logic (adjust these based on your data)
    tamper_threshold_std = 2500  # example value — tune based on your dataset
    authentic_sources = ['Original', 'Authentic']
    known_scanners = [label for label in le.classes_ if label not in ['Tampered'] + authentic_sources]

    if fft_std > tamper_threshold_std:
        final_label = 'Tampered'
    elif prediction_label in authentic_sources:
        final_label = 'Authentic'
    elif prediction_label in known_scanners:
        final_label = 'Original'
    else:
        final_label = 'Authentic'  # fallback if label is unknown but FFT looks clean

    # Display result
    st.markdown(f"### 🧠 Prediction: **{final_label}**")

    # Prepare result for download
    result_df = pd.DataFrame({
        "Filename": [uploaded_file.name],
        "Prediction": [final_label],
        "FFT Mean": [fft_mean],
        "FFT Std": [fft_std]
    })

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Result as CSV",
        data=csv,
        file_name="tracefinder_result.csv",
        mime="text/csv"
    )
