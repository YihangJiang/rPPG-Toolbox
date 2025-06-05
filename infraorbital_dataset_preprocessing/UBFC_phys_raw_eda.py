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

# Convert list to 2D numpy array (shape: num_signals x signal_length)
group_of_signals = np.array(t3_bvp_list)

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
from e2epyppg.utils import get_data
from e2epyppg.ppg_sqa import sqa

# %%
# Provide your PPG signal and sampling rate (you can use your own signal in format `np.ndarray`)
t1_list = []
t2_list = []
t3_list = []
for i, file in enumerate(csv_files):
    print(i, file)
    # Load signal
    df = pd.read_csv(file, header=None)
    bvp_signal = df.iloc[:, 0].dropna().values

    input_sig = bvp_signal

    sampling_rate = 64

    # Set this parameter True if the signal has not been filtered:
    filter_signal = True

    # Call the PPG signal quality assessment function
    clean_indices, noisy_indices = sqa(input_sig, sampling_rate, filter_signal)
    if 'T1' in file:
        if len(clean_indices)>0:
            t1_list.append(len(clean_indices[0])/len(noisy_indices[0]))
        else:
            t1_list.append(0)
    elif 'T2' in file:
        if len(clean_indices)>0:
            t2_list.append(len(clean_indices[0])/len(noisy_indices[0]))
        else:
            t2_list.append(0)
    else:
        if len(clean_indices)>0:
            t3_list.append(len(clean_indices[0])/len(noisy_indices[0]))
        else:
            t3_list.append(0)


# %%

import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# Example PPG signal (replace with your actual 11520-sample signal)
# signal = your actual signal
signal = np.random.randn(11520)  # placeholder

fs = 64  # Sampling frequency (Hz)
window_size = 8 * fs  # 8 seconds = 512 samples
step_size = window_size // 2  # 50% overlap = 256 samples

heart_rates = []
timestamps = []

for start in range(0, len(signal) - window_size + 1, step_size):
    window = signal[start:start + window_size]
    
    try:
        # Preprocess and detect peaks
        cleaned = nk.ppg_clean(window, sampling_rate=fs)
        peaks, _ = nk.ppg_peaks(cleaned, sampling_rate=fs)

        # Calculate heart rate from detected peaks
        rpeaks = np.where(peaks["PPG_Peaks"] == 1)[0]
        if len(rpeaks) >= 2:
            ibi = np.diff(rpeaks) / fs  # Inter-beat intervals in seconds
            hr = 60 / np.mean(ibi)  # Convert to BPM
        else:
            hr = np.nan  # Not enough peaks
    except Exception:
        hr = np.nan

    heart_rates.append(hr)
    timestamps.append(start / fs)

# Plotting the heart rate trend
plt.plot(timestamps, heart_rates, marker='o')
plt.xlabel("Time (s)")
plt.ylabel("Heart Rate (BPM)")
plt.title("Heart Rate Over Time (Sliding Window)")
plt.grid(True)
plt.show()

# %%
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# Example PPG signal (replace with your actual 11520-sample signal)
# signal = your actual signal
heart_rates = []
for signal in t1_bvp_list:
    fs = 64  # Sampling frequency (Hz)
    window_size = 8 * fs  # 8 seconds = 512 samples
    step_size = window_size // 2  # 50% overlap = 256 samples

    timestamps = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        
        try:
            # Preprocess and detect peaks
            cleaned = nk.ppg_clean(window, sampling_rate=fs)
            peaks, _ = nk.ppg_peaks(window, sampling_rate=fs)

            # Calculate heart rate from detected peaks
            rpeaks = np.where(peaks["PPG_Peaks"] == 1)[0]
            if len(rpeaks) >= 2:
                ibi = np.diff(rpeaks) / fs  # Inter-beat intervals in seconds
                hr = 60 / np.mean(ibi)  # Convert to BPM
            else:
                hr = np.nan  # Not enough peaks
        except Exception:
            hr = np.nan

        heart_rates.append(hr)
        timestamps.append(start / fs)

plt.figure(figsize=(8, 5))
plt.hist(heart_rates, bins=180, edgecolor='black')
plt.xlabel("Heart Rate (bpm)")
plt.ylabel("Frequency")
plt.xlim(20,200)
plt.title(f"Heart Rate Distribution neurokit")
plt.tight_layout()
plt.show()

# %%

import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# Example PPG signal (replace with your actual 11520-sample signal)
# signal = your actual signal
heart_rates = []
for signal in t1_bvp_list:
    fs = 64  # Sampling frequency (Hz)
    window_size = 8 * fs  # 8 seconds = 512 samples
    step_size = window_size // 2  # 50% overlap = 256 samples

    timestamps = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        
        try:
            wd, m = hp.process(window, sample_rate=fs)
            hr = m['bpm']  # Heart rate in BPM
        except:
            hr = np.nan  # Handle bad segments

        heart_rates.append(hr)
        timestamps.append(start / fs)

plt.figure(figsize=(8, 5))
plt.hist(heart_rates, bins=180, edgecolor='black')
plt.xlabel("Heart Rate (bpm)")
plt.ylabel("Frequency")
plt.xlim(20,200)
plt.title(f"Heart Rate Distribution HeartPy")
plt.tight_layout()
plt.show()
# %%

# %%
heart_rates = []
for signal in t1_bvp_list:
    fs = 64  # Sampling frequency (Hz)
    window_size = 8 * fs  # 8 seconds = 512 samples
    step_size = window_size // 2  # 50% overlap = 256 samples

    timestamps = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        
        try:
            hr_mean, _, _ = estimate_hr_fft(window, fs=bvp_fs)
        except:
            hr = np.nan  # Handle bad segments

        heart_rates.append(hr_mean)
        timestamps.append(start / fs)

plt.figure(figsize=(8, 5))
plt.hist(heart_rates, bins=180, edgecolor='black')
plt.xlabel("Heart Rate (bpm)")
plt.ylabel("Frequency")
plt.xlim(20,200)
plt.title(f"Heart Rate Distribution FFT")
plt.tight_layout()
plt.show()

# %%
