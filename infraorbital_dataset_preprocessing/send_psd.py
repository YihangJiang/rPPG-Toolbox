# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import periodogram
import cv2
import re
import numpy as np

# Directory containing the CSV and video files
test_data_dir = '/work/yj167/DATASET_1/'

# Regular expression to match subfolders with the format "s<number>"
subfolder_pattern = re.compile(r's\d+')

# Iterate through subfolders
csv_files = []
vid_files = []
for root, dirs, files in os.walk(test_data_dir):
    # Only consider directories that match the pattern
    if re.search(subfolder_pattern, os.path.basename(root)):
        csv_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv') and "bvp" in f.lower()])
        vid_files.extend([os.path.join(root, f) for f in files if f.endswith('.avi')])

# %%
vid_fs = 35  # Hz
bvp_fs = 64
heartpy = False 
chunk_size = 640  # frames (samples)
chunked_heart_rates = []
t1_bvp_list, t2_bvp_list, t3_bvp_list = [], [], []

for file in csv_files:
    try:
        # Load signal
        df = pd.read_csv(file, header=None)
        bvp_signal = df.iloc[:, 0].dropna().values
        if 'T1' in file:
            t1_bvp_list.append(bvp_signal)
        elif 'T2' in file:
            t2_bvp_list.append(bvp_signal)
        else:
            t3_bvp_list.append(bvp_signal)
    
    except Exception as e:
        print(f"Error in {file}: {e}")
        continue

# %%
# Convert list to 2D numpy array (shape: num_signals x signal_length)
group_of_signals = np.array(t1_bvp_list)

# Compute PSD using Welch's method
frequencies, psd_group = welch(group_of_signals, fs=bvp_fs, nperseg=256, axis=1)

# Average PSD across signals
mean_psd = np.mean(psd_group, axis=0)

# Plot the average PSD
plt.figure(figsize=(10, 6))
plt.plot(frequencies, mean_psd)
plt.title("Average Power Spectral Density (resting)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.xlim(0.5, 5)
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
