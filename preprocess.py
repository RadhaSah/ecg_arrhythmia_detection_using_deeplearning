import wfdb
import os
import numpy as np

# ---------------------------
# Paths
# ---------------------------
data_dir = "data/mitdb"             # MIT-BIH downloaded folder
output_file = "processed_data.npz"  # Output for training

os.makedirs(data_dir, exist_ok=True)

# ---------------------------
# Download MIT-BIH (if not already)
# ---------------------------
if not any(fname.endswith(".dat") for fname in os.listdir(data_dir)):
    print("⏳ Downloading MIT-BIH Arrhythmia Database...")
    wfdb.dl_database("mitdb", dl_dir=data_dir)
    print(f"✅ Download complete! Files are saved in: {data_dir}")

# ---------------------------
# Process each record
# ---------------------------
X, y = [], []

for record_file in os.listdir(data_dir):
    if record_file.endswith(".dat"):
        record_name = record_file[:-4]
        sig, fields = wfdb.rdsamp(os.path.join(data_dir, record_name))
        ann = wfdb.rdann(os.path.join(data_dir, record_name), "atr")

        # Simple segmentation: 180 samples per beat around R-peak
        for r in ann.sample:
            start = r - 90
            end = r + 90
            if start >= 0 and end <= len(sig):
                beat = sig[start:end, 0]  # first channel only
                X.append(beat)
                y.append(0)  # dummy label (abhi 0, baad me class mapping kar sakte ho)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Save as npz
np.savez(output_file, X=X, y=y)
print(f"✅ Processed dataset saved to {output_file}")
print(f"Total beats: {len(X)}")
