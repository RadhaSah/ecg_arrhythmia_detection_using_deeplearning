import os
import wfdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample
import json

# ---------------------------
# Config
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")  # base data folder
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

# Label map
label_map = {
    "0": "Normal",
    "1": "Atrial Premature Beat (APB)",
    "2": "Premature Ventricular Contraction (PVC)",
    "3": "Fusion of Ventricular and Normal Beat",
    "4": "Unclassifiable Beat"
}

with open(os.path.join(BASE_DIR, "labels.json"), "w") as f:
    json.dump(label_map, f, indent=2)

# ---------------------------
# Dataset
# ---------------------------
class ECGDataset(Dataset):
    def __init__(self, data_dir):
        self.beats = []
        self.labels = []

        # Auto-detect subfolders
        subfolders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if not subfolders:
            subfolders = [data_dir]  # if no subfolder, use base folder

        print("🔍 Scanning folders:", subfolders)

        for folder in subfolders:
            records = [os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(".hea")]
            print(f"📂 Found records in {folder}:", records)

            for rec in records:
                try:
                    record = wfdb.rdrecord(os.path.join(folder, rec))
                    ann = wfdb.rdann(os.path.join(folder, rec), "atr")
                    signal = record.p_signal[:, 0]

                    for r, sym in zip(ann.sample, ann.symbol):
                        start, end = max(0, r-90), min(len(signal), r+90)
                        beat = resample(signal[start:end], 180)

                        # Map annotation symbol to class
                        if sym == "N":   label = 0
                        elif sym == "A": label = 1
                        elif sym == "V": label = 2
                        elif sym == "F": label = 3
                        else:             label = 4

                        self.beats.append(beat)
                        self.labels.append(label)

                except Exception as e:
                    print(f"⚠️ Error reading record {rec}: {e}")

        print(f"✅ Total beats loaded: {len(self.beats)}")

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, idx):
        x = torch.tensor(self.beats[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------------------------
# Model
# ---------------------------
class ECGNet(nn.Module):
    def __init__(self):
        super(ECGNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)

        dummy = torch.zeros(1, 1, 180)
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool2(F.relu(self.conv3(x)))
            num_features = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, len(label_map))

    def forward(self, x):
        x1 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x3 = self.pool2(F.relu(self.conv3(x2)))
        x_flat = x3.view(x3.size(0), -1)
        out = F.relu(self.fc1(x_flat))
        out = self.fc2(out)
        return out


# ---------------------------
# Training
# ---------------------------
if __name__ == "__main__":
    dataset = ECGDataset(DATA_DIR)
    if len(dataset) == 0:
        raise RuntimeError("❌ No beats loaded. Check your dataset folder and annotation symbols.")

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ECGNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    print("✅ Training complete. Model saved at outputs/model.pt")
