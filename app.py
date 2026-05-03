import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
import json
import os
import matplotlib.pyplot as plt
import tempfile


# Optional WFDB support
try:
    import wfdb
    WFDB_AVAILABLE = True
except:
    WFDB_AVAILABLE = False

# ---------------------------
# Load Label Map
# ---------------------------
base_dir = os.path.dirname(__file__)
label_file = os.path.join(base_dir, "labels.json")
with open(label_file) as f:
    label_map = json.load(f)

# ---------------------------
# Model definition
# ---------------------------
class ECGNet(nn.Module):
    def __init__(self):
        super(ECGNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)

        dummy = torch.zeros(1,1,180)
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool2(F.relu(self.conv3(x)))
            num_features = x.view(1,-1).size(1)

        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, len(label_map))

    def forward(self, x):
        x1 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(F.relu(self.conv2(x1)))
        x3 = self.pool2(F.relu(self.conv3(x2)))  # last conv features
        x_flat = x3.view(x3.size(0), -1)
        out = F.relu(self.fc1(x_flat))
        out = self.fc2(out)
        return out, x3

# ---------------------------
# Load pretrained model
# ---------------------------
model = ECGNet()
model_path = os.path.join(base_dir, "outputs", "model.pt")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ---------------------------
# Streamlit UI config
# ---------------------------
st.set_page_config(page_title="🫀 ECG Arrhythmia Detection", layout="wide")

st.sidebar.header("⚙️ Settings")
# ---------------------------
# 💬 Offline ECG Assistant
# ---------------------------
st.sidebar.subheader("💬 ECG Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.markdown("### 💡 Try asking:")
st.sidebar.markdown("""
- How CNN works in ECG?
- Difference between PVC and APB?
- Why model gives uncertain?
- Explain arrhythmia in detail
""")

user_input = st.sidebar.text_input("Ask your question:")


def chatbot_response(query):
    q = query.lower()

    # Simple
    if "ecg" in q:
        return "ECG (Electrocardiogram) measures electrical activity of the heart."

    elif "arrhythmia" in q:
        return "Arrhythmia is an irregular heartbeat caused by abnormal electrical signals."

    elif "pvc" in q:
        return "PVC is an early heartbeat from ventricles, often showing a wide waveform."

    elif "apb" in q:
        return "APB is an early heartbeat from atria."

    # Deep
    elif "cnn" in q:
        return ("CNN automatically extracts features like QRS complex, P waves, and T waves "
                "from ECG signals and classifies them.")

    elif "difference between pvc and apb" in q:
        return ("APB originates in atria, while PVC originates in ventricles. "
                "PVC has wider abnormal waveform.")

    elif "uncertain" in q:
        return ("Uncertain is used when model confidence is low to avoid incorrect predictions.")

    elif "explain arrhythmia" in q:
        return ("Arrhythmia is abnormal heart rhythm. Types include APB, PVC, tachycardia, and bradycardia.")

    elif "risk" in q:
        return "Risk is calculated based on abnormal beats percentage: Low (<30%), Moderate (30–60%), High (>60%)."

    elif "dataset" in q:
        return "The model is trained on MIT-BIH Arrhythmia dataset."

    elif "model working" in q:
        return ("ECG signals are processed, beats extracted, CNN classifies them, and risk is calculated.")

    # Fallback
    else:
        return "Sorry, I can answer ECG-related questions only (try CNN, PVC, APB, arrhythmia, risk, model)."


# Handle input
if user_input:
    response = chatbot_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))


# Display chat
for role, msg in st.session_state.chat_history:
    st.sidebar.write(f"**{role}:** {msg}")

dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)  # Default = Light mode

# Optional plotting theme
if dark_mode:
    plt.style.use("dark_background")
    st.markdown("<style>body {background-color: #00FFF7; color: white;}</style>", unsafe_allow_html=True)
else:
    plt.style.use("default")

st.title("🫀 ECG Arrhythmia Detection with Grad-CAM")
st.caption("Upload ECG (WFDB or CSV). Beat-wise predictions, confidence, and final summary shown below.")

uploaded_files = st.file_uploader(
    "Upload WFDB files (.dat/.hea/.atr) or CSV",
    type=["dat","hea","atr","csv"],
    accept_multiple_files=True
)

# ---------------------------
# Plot function
# ---------------------------
def plot_gradcam(signal, cam, idx, label, conf):
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(signal, color="white" if dark_mode else "black", label="ECG")
    cam_resized = resample(cam, len(signal))
    cam_norm = (cam_resized - cam_resized.min())/(cam_resized.max()-cam_resized.min()+1e-8)
    ax.fill_between(np.arange(len(signal)),
                    min(signal), max(signal),
                    where=cam_norm>0.3, alpha=0.3, color="red", label="Grad-CAM")
    ax.set_title(f"Beat {idx} → {label} ({conf:.1f}%)", color="white" if dark_mode else "black")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------
# WFDB loader
# ---------------------------
def load_wfdb(uploaded_files):
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in uploaded_files:
            with open(os.path.join(tmpdir, f.name), "wb") as fh:
                fh.write(f.getvalue())
        base = os.path.splitext(uploaded_files[0].name)[0]
        record = wfdb.rdrecord(os.path.join(tmpdir, base))
        ann = wfdb.rdann(os.path.join(tmpdir, base), "atr")
        return record.p_signal[:,0], record.fs, ann.sample

# ---------------------------
# Main processing
# ---------------------------
signal, fs, r_peaks = None, None, None
if uploaded_files:
    wfdb_files = [f for f in uploaded_files if f.name.endswith((".dat",".hea",".atr"))]
    csv_files = [f for f in uploaded_files if f.name.endswith(".csv")]
    if wfdb_files and WFDB_AVAILABLE:
        try:
            signal, fs, r_peaks = load_wfdb(wfdb_files)
            st.success(f"✅ Loaded WFDB signal (len={len(signal)}, fs={fs} Hz)")
            st.info(f"Found {len(r_peaks)} beats from annotation")
        except Exception as e:
            st.error(f"Error reading WFDB: {e}")
    elif csv_files:
        df = pd.read_csv(csv_files[0])

    
        signal = pd.to_numeric(df.iloc[:,0], errors='coerce')
        signal = signal.dropna().values

        st.success(f"✅ Loaded CSV signal (len={len(signal)})")
    else:
        st.warning("Please upload WFDB (.dat/.hea/.atr) or CSV")

if signal is not None:
    max_beats = st.slider("Number of beats to analyze", 5, 50, 20)
    preds, confidences, labels = [], [], []

    if r_peaks is not None:
        beats_idx = r_peaks[:max_beats]
        beats = []
        for r in beats_idx:
            start, end = max(0, r-90), min(len(signal), r+90)
            beat = resample(signal[start:end], 180)
            beat = (beat - np.mean(beat)) / (np.std(beat)+1e-8)
            beats.append(beat)
    else:
        sections = max(1, len(signal)//180)
        beats = [resample(b, 180) for b in np.array_split(signal, sections)[:max_beats]]
        beats = [(b - np.mean(b))/(np.std(b)+1e-8) for b in beats]

    # Process each beat
    for i, beat in enumerate(beats):
        tensor = torch.tensor(beat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor.requires_grad = True
        grad_holder = {}
        def save_grad(grad): grad_holder["grad"]=grad

        out, features = model(tensor)

        # Probability and confidence
        probs = torch.softmax(out, dim=1).detach().numpy()[0]
        pred = int(np.argmax(probs))
        confidence = float(probs[pred] * 100.0)

        preds.append(pred)
        confidences.append(confidence)
        if confidence < 60:
            label = "Uncertain"
        else:
            label = label_map[str(pred)]
        labels.append(label)
        
        # Grad-CAM calculation
        h = features.register_hook(save_grad)
        model.zero_grad()
        out[0,pred].backward(retain_graph=True)
        grads = grad_holder["grad"]
        weights = torch.mean(grads, dim=2, keepdim=True)
        cam = torch.sum(weights * features, dim=1).squeeze().detach().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        h.remove()
        plot_gradcam(beat, cam, i, label, confidence)

    # ---------------------------
    # Beat-wise Prediction Table
    # ---------------------------
    st.subheader("📊 Beat-wise Predictions")
    df_out = pd.DataFrame({
        "Beat": list(range(len(preds))),
        "Prediction Index": preds,
        "Label": labels,
        "Confidence": confidences
    })

    df_display = df_out.copy()
    df_display["Confidence (%)"] = df_display["Confidence"].map(lambda x: f"{x:.2f}%")
    df_display = df_display[["Beat", "Prediction Index", "Label", "Confidence (%)"]]

    st.dataframe(df_display, use_container_width=True)

    st.download_button("⬇️ Download Predictions CSV",
                       df_out.to_csv(index=False).encode("utf-8"),
                       "predictions.csv","text/csv")

    # ---------------------------
    # Detailed Final Summary
    # ---------------------------
    st.subheader("🩺 Final Diagnosis")

    disease_beats = df_out[
    (df_out["Label"] != "Normal") &
    (df_out["Label"] != "Unclassifiable Beat") &
    (df_out["Label"] != "Uncertain")]
    total_beats = len(df_out)

    if len(disease_beats) == 0:
        st.success(f"✅ Mostly Normal - No disease detected in {total_beats} beats")
    else:
        disease_count = len(disease_beats)
        disease_percentage = (disease_count/total_beats)*100
        st.error(f"⚠️ Disease detected in {disease_count}/{total_beats} beats ({disease_percentage:.1f}%)")
        disease_summary = disease_beats["Label"].value_counts()
        # Optional safety filter
        disease_summary = disease_summary.drop(labels=["Uncertain"], errors='ignore')
        st.table(disease_summary)

        # Risk Level
        if disease_percentage > 60:
           st.error("🔴 High Risk Detected")
        elif disease_percentage > 30:
           st.warning("🟠 Moderate Risk")
        else:
           st.success("🟢 Low Risk")
        st.subheader("Possible Conditions Explanation")

        disease_info = {
    "Atrial Premature Beat (APB)": "Irregular early beats from atria. May indicate atrial arrhythmia.",
    "Fusion of Ventricular and Normal Beat": "Abnormal fusion of normal and ventricular beats. Can indicate ventricular issues.",
    "Unclassifiable Beat": "Irregular pattern not clearly classified. Needs further clinical analysis.",
    "Normal": "No abnormality detected.",
    "Uncertain": "Model is not confident about this beat. May be noise or unclear signal."}

        for disease, count in disease_summary.items():
            if disease in disease_info:
                st.markdown(f"### 🔹 {disease}")
                st.write(f"**Explanation:** {disease_info[disease]}")
                st.write(f" **Detected in {count} beats**") 
        st.warning("⚠️ This is an AI-based prediction. Please consult a medical professional.")
