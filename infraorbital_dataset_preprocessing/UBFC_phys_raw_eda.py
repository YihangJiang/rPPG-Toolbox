# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import periodogram
import cv2
import re
import heartpy as hp

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
# Initialize list to store number of rows in each CSV file
bvp_counts = []

# Count the number of rows in each file
for file in csv_files:
    try:
        df = pd.read_csv(file, header=None)
        bvp_counts.append(len(df))
    except Exception as e:
        bvp_counts.append(0)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(bvp_counts, bins=1, edgecolor='black')
plt.xlabel("Number of BVP Readings")
plt.ylabel("Frequency")
plt.title("Histogram of BVP Reading Counts Across Files")
plt.xticks(sorted(set(bvp_counts)))  # Show only unique frame counts
plt.tight_layout()
plt.show()

# %%
# Initialize list to store frame counts
frame_counts = []

# Loop through each file and get the frame count
for file in vid_files:
    try:
        cap = cv2.VideoCapture(file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counts.append(frame_count)
        cap.release()
    except Exception as e:
        frame_counts.append(0)

# Plot histogram of frame counts
plt.figure(figsize=(8, 5))
plt.hist(frame_counts, bins=2, edgecolor='black')
plt.xlabel("Number of Frames")
plt.ylabel("Frequency")
plt.title("Histogram of Frame Counts Across Whole Face Video Files")
plt.xticks(sorted(set(frame_counts)))  # Show only unique frame counts
plt.tight_layout()
plt.show()

# %%
def estimate_hr_fft(bvp_signal, fs):
    """Estimates heart rate and returns spectrum for plotting."""
    N = len(bvp_signal)
    bvp_detrended = detrend(bvp_signal)
    freqs = np.fft.fftfreq(N, d=1/fs)
    fft_spectrum = np.abs(fft(bvp_detrended))**2

    # Filter to positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_spectrum = fft_spectrum[pos_mask]

    # Limit to HR band
    hr_mask = (freqs >= 0.5) & (freqs <= 5.0)
    freqs_hr = freqs[hr_mask]
    spectrum_hr = fft_spectrum[hr_mask]

    if len(freqs_hr) == 0:
        return None, None, None

    peak_idx = np.argmax(spectrum_hr)
    peak_freq = freqs_hr[peak_idx]
    estimated_hr_bpm = peak_freq * 60

    return estimated_hr_bpm, freqs_hr * 60, spectrum_hr  # HR in BPM

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.fftpack import fft


vid_fs = 35  # Hz
bvp_fs = 64
heartpy = True
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

        # Divide into chunks of 600 samples
        num_chunks = len(bvp_signal) // chunk_size

        for i in range(num_chunks):
            chunk = bvp_signal[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk) < chunk_size:
                continue  # Skip incomplete chunk

            # Process chunk
            if heartpy:
                hp.process(chunk, bvp_fs)
            hr_mean, _, _ = estimate_hr_fft(chunk, fs=bvp_fs)

            # Store HR if it's valid
            if not np.isnan(hr_mean):
                chunked_heart_rates.append(hr_mean)

    except Exception as e:
        print(f"Error in {file}: {e}")
        continue

# Plot histogram of chunked heart rates
plt.figure(figsize=(8, 5))
plt.hist(chunked_heart_rates, bins='auto', edgecolor='black')
plt.xlabel("Heart Rate (bpm)")
plt.ylabel("Frequency")
plt.xlim(20,200)
plt.title(f"Heart Rate Distribution from BVP signals ({chunk_size}-Sampling Chunks / {chunk_size / bvp_fs} seconds)")
plt.tight_layout()
plt.show()

# %%

# Example: list of BVP signals (each signal is a 1D numpy array)
# Replace this with loading your actual BVP data

fs = 64  # sampling frequency in Hz (adjust to your BVP data's actual rate)

plt.figure(figsize=(12, 6))
f_filtered_list = []
Pxx_filtered_list = []

for i, bvp in enumerate(t3_bvp_list):
    f, Pxx = periodogram(bvp, fs=fs)

    freq_limit = (f >= 0.1) & (f <= 5)
    f_filtered = f[freq_limit]
    Pxx_filtered = Pxx[freq_limit]
    f_filtered_list.append(f_filtered)
    Pxx_filtered_list.append(Pxx_filtered)
    plt.semilogy(f_filtered, Pxx_filtered)

plt.title("Periodogram of BVP Signals T3")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Example PPG signal (replace this with your actual signal)
# For demonstration, we generate a synthetic PPG signal with a sampling rate of 100 Hz
sampling_rate = 100  # 100 Hz
t = np.linspace(0, 10, 10 * sampling_rate)  # 10 seconds of data
signal = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.random.randn(t.size)  # 1 Hz signal with noise

# Process the signal
processed_data, heart_rate = hp.process(signal, sampling_rate)

# Print heart rate
print(f"Heart rate: {heart_rate['BPM']} BPM")

# Visualize the processed signal and the detected peaks
hp.plotter(processed_data)

# %%
import heartpy as hp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# List to store the heart rates from all signals
heart_rates = []

# Assuming signals is a list of 50 PPG signals, replace with actual data
# For demonstration, let's simulate 50 PPG signals
# Each signal has 1000 samples and a sampling rate of 100 Hz
sampling_rate = 100  # 100 Hz
num_signals = 50
signal_length = 1000
signals = [np.sin(2 * np.pi * 1.0 * np.linspace(0, 10, signal_length)) + 0.5 * np.random.randn(signal_length) for _ in range(num_signals)]

# Process each signal and calculate heart rate
for signal in signals:
    processed_data, heart_rate = hp.process(signal, sampling_rate)
    heart_rates.append(heart_rate['BPM'])

# Plot histogram of heart rates to visualize the most frequent heart rate
plt.figure(figsize=(10, 6))
plt.hist(heart_rates, bins=10, edgecolor='black')
plt.title('Distribution of Heart Rates Across 50 Signals')
plt.xlabel('Heart Rate (BPM)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Optionally, you can also calculate the most frequent heart rate
most_frequent_hr = pd.Series(heart_rates).mode()[0]
print(f"The most frequent heart rate is {most_frequent_hr:.2f} BPM")


# %%
import numpy as np
import matplotlib.pyplot as plt

# Convert list to 2D numpy array (shape: num_signals x signal_length)
group_of_signals = np.array(t2_bvp_list)

# Compute PSD using Welch's method
frequencies, psd_group = welch(group_of_signals, fs=bvp_fs, nperseg=256, axis=1)

# Average PSD across signals
mean_psd = np.mean(psd_group, axis=0)

# Plot the average PSD
plt.figure(figsize=(10, 6))
plt.plot(frequencies, mean_psd)
plt.title("Average Power Spectral Density (arithmetic)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.xlim(0.5, 5)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
