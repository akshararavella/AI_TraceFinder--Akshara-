from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Simulate patch-level features
X_patch = np.random.rand(500, 30)  # 500 patches, 30 features

scaler_patch = StandardScaler()
scaler_patch.fit(X_patch)

joblib.dump(scaler_patch, "patch_scaler.pkl")
