# ecg_project
ECG Arrhythmia Detection using Deep Learning : A deep learning model using CNN/LSTM to classify ECG signals into normal and abnormal heart rhythms. Trained on standard datasets, it enables accurate and automated arrhythmia detection, reducing manual effort and supporting early diagnosis.


📂 Dataset Setup

This project does not include the dataset in the repository due to size and licensing restrictions. Please follow the steps below to set it up locally.

1. Download Dataset

Download the MIT-BIH Arrhythmia Dataset from:

https://physionet.org/content/mitdb/
2. Create Folder Structure

Inside the project directory, create the following folder:

project-root/
│── data/
│   └── (place dataset files here)
│── src/
│── README.md
3. Add Dataset Files

Extract and place all downloaded dataset files inside the data/ folder.

4. Update File Paths (if needed)

Ensure your code points to the correct dataset path. Example:

data_path = "data/"
5. Verify Setup

Run the preprocessing or training script to confirm the dataset is loaded correctly.