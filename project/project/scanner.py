import pickle
import numpy as np

# Simulate residuals for 3 scanners
scanner_fps = {
    "Canon120": np.random.rand(256, 256),
    "EpsonV600": np.random.rand(256, 256),
    "HPScanJet": np.random.rand(256, 256)
}

# Save fingerprints
with open("scannerfingerprints.pkl", "wb") as f:
    pickle.dump(scanner_fps, f)

# Save keys separately
np.save("fp_keys.npy", list(scanner_fps.keys()))
