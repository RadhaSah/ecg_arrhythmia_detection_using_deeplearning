import wfdb
import os

# dataset save hone wali jagah
save_dir = "data/mitdb"
os.makedirs(save_dir, exist_ok=True)

print("⏳ Downloading MIT-BIH Arrhythmia Database...")
wfdb.dl_database("mitdb", dl_dir=save_dir)

print(f"✅ Download complete! Files are saved in: {save_dir}")
