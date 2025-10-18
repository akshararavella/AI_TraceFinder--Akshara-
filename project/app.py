import streamlit as st
from PIL import Image
import numpy as np
import joblib
import pickle
import json
import cv2
import pywt
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd

# Load models and scalers
model = joblib.load("models/LGBM_model_final.joblib")
le = joblib.load("models/LGBM_label_encoder_final.joblib")
scaler = joblib.load("models/hybrid_feat_scaler.pkl")
patch_scaler = joblib.load("models/artifacts_tamper_patch/patch_scaler.pkl")
patch_svm = joblib.load("models/artifacts_tamper_patch/patch_svm_sig_calibrated.pkl")
with open("models/scannerfingerprints.pkl", "rb") as f:
    scanner_fps = pickle.load(f)
fp_keys = np.load("models/fp_keys.npy", allow_pickle=True)
with open("models/artifacts_tamper_patch/thresholds_patch.json", "r") as f:
    patch_thresholds = json.load(f)

# Feature extraction functions
def extract_fft_features(image):
    img_np = np.array(image)
    fft = np.fft.fft2(img_np)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    fft_mean = np.mean(magnitude)
    fft_std = np.std(magnitude)
    return [fft_mean, fft_std]

def extract_lbp_features(image):
    img_np = np.array(image)
    lbp = local_binary_pattern(img_np, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return hist.tolist()

def extract_wavelet_residual(image):
    img_np = np.array(image)
    coeffs2 = pywt.dwt2(img_np, 'haar')
    _, (LH, HL, HH) = coeffs2
    residual = np.abs(LH) + np.abs(HL) + np.abs(HH)
    return residual

def extract_radial_energy(image):
    img_np = np.array(image)
    fft = np.fft.fft2(img_np)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    center = tuple(np.array(magnitude.shape) // 2)
    radial_profile = []
    for r in range(1, min(center)):
        mask = np.zeros_like(magnitude)
        cv2.circle(mask, center[::-1], r, 1, thickness=1)
        radial_profile.append(np.mean(magnitude[mask == 1]))
    return radial_profile[:20]

def extract_hybrid_features(image):
    image = image.convert('L').resize((256, 256))
    fft_feat = extract_fft_features(image)
    lbp_feat = extract_lbp_features(image)
    radial_feat = extract_radial_energy(image)
    return fft_feat + lbp_feat + radial_feat

def match_scanner_fingerprint(image):
    residual = extract_wavelet_residual(image.resize((256, 256)).convert('L'))
    residual_flat = residual.flatten().reshape(1, -1)
    best_match = None
    best_score = -1
    for key in fp_keys:
        ref = scanner_fps[key].flatten().reshape(1, -1)
        score = cosine_similarity(residual_flat, ref)[0][0]
        if score > best_score:
            best_score = score
            best_match = key
    return best_match, best_score

def extract_patch_features(image):
    image = image.convert('L').resize((256, 256))
    img_np = np.array(image)
    patch_size = 64
    features = []
    for i in range(0, 256, patch_size):
        for j in range(0, 256, patch_size):
            patch = img_np[i:i+patch_size, j:j+patch_size]
            fft = np.fft.fft2(patch)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            fft_mean = np.mean(magnitude)
            fft_std = np.std(magnitude)
            lbp = local_binary_pattern(patch, P=8, R=1, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
            features.append([fft_mean, fft_std] + hist.tolist())
    return np.array(features)

def detect_tampering(image, scanner_label):
    patch_feats = extract_patch_features(image)
    patch_feats_scaled = patch_scaler.transform(patch_feats)
    probs = patch_svm.predict_proba(patch_feats_scaled)[:, 1]
    tamper_ratio = np.mean(probs > patch_thresholds.get(scanner_label, 0.5))
    return tamper_ratio > 0.3, tamper_ratio

# Streamlit UI
st.set_page_config(page_title="🔍 TraceFinder", page_icon="🔍", layout="wide")
st.markdown(
    """
    <div style='text-align:center; padding-top:16px;'>
        <h1>🔍 TraceFinder</h1>
        <h4 style='color:#6a7ff7;'>Advanced Scanner Identification & Tamper Detection 🖨️</h4>
        <p style='color:#aaaaff; font-size:20px;'>Upload a scanned image below 👇</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("📁 Upload scanned image (PNG/JPG/JPEG/TIF/TIFF)", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Hybrid features and prediction
    hybrid_feat = extract_hybrid_features(image)
    hybrid_scaled = scaler.transform([hybrid_feat])
    probs = model.predict_proba(hybrid_scaled)[0]
    max_prob = np.max(probs)
    prediction_idx = np.argmax(probs)
    prediction_label = le.inverse_transform([prediction_idx])[0]

    # Fingerprint matching
    matched_scanner, match_score = match_scanner_fingerprint(image)

    # Tamper detection
    is_tampered, tamper_score = detect_tampering(image, prediction_label)

    # Final verdict
    if is_tampered:
        display_label = "Tampered"
        color = "#FF4B4B"
    elif matched_scanner:
        display_label = f"Flatfield Scanner ({matched_scanner})"
        color = "#B36BFF"
    else:
        display_label = "Authentic"
        color = "#4CAF50"

    # Display results
    st.markdown("## 🔍 Analysis Result")
    st.markdown(
        f"""
        <div style='padding:24px;border-radius:12px;background:#181929;border:2px solid {color};'>
            <div style='font-size:22px;color:{color};'>🖨️ Verdict</div>
            <div style='font-size:32px;margin-top:10px;font-weight:bold;color:{color};'>{display_label}</div>
            <div style='font-size:16px;color:#d7a6ff;margin-top:14px;'>🎯 Model Confidence: <b>{max_prob * 100:.2f}%</b></div>
            <div style='font-size:16px;color:#d7a6ff;'>🧬 Tamper Score: <b>{tamper_score:.2f}</b></div>
            <div style='font-size:16px;color:#d7a6ff;'>📌 Fingerprint Match: <b>{matched_scanner} ({match_score:.2f})</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Debug panel
    st.markdown("### 🧪 Raw Model Output")
    st.write("Class probabilities:", dict(zip(le.classes_, [round(p * 100, 2) for p in probs])))

    # CSV download
    result_df = pd.DataFrame({
        "File": [uploaded_file.name],
        "Verdict": [display_label],
        "Model Confidence (%)": [round(max_prob * 100, 2)],
        "Tamper Score": [round(tamper_score, 2)],
        "Fingerprint Match": [matched_scanner],
        "Match Score": [round(match_score, 2)]
    })

    st.download_button(
        label="📥 Download Results as CSV",
        data=result_df.to_csv(index=False),
        file_name="tracefinder_results.csv",
        mime="text/csv"
    )
else:
    st.info("💡 Drag-and-drop or browse to upload and analyze.")
