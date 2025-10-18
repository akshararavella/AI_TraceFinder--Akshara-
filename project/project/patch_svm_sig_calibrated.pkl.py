from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
import numpy as np

# Simulate patch features and labels
X_patch = np.random.rand(500, 30)
y_patch = np.random.randint(0, 2, size=500)

svm = SVC(probability=True)
calibrated_svm = CalibratedClassifierCV(svm, cv=5)
calibrated_svm.fit(X_patch, y_patch)

joblib.dump(calibrated_svm, "patch_svm_sig_calibrated.pkl")
