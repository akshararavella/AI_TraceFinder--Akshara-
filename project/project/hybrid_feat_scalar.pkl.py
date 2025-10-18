from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Simulate feature matrix (replace with real features)
X = np.random.rand(100, 20)  # 100 samples, 20 features

scaler = StandardScaler()
scaler.fit(X)

joblib.dump(scaler, "hybrid_feat_scaler.pkl")




#your_project/
#├── app.py
#├── models/
#│   ├── LGBM_model_final.joblib
#│   ├── LGBM_label_encoder_final.joblib
#│   ├── hybrid_feat_scaler.pkl
#│   ├── scannerfingerprints.pkl
#│   ├── fp_keys.npy
#│   └── artifacts_tamper_patch/
#│       ├── patch_svm_sig_calibrated.pkl
#│       ├── patch_scaler.pkl
#│       └── thresholds_patch.json
