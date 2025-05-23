# %%
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import os

def read_bvp_from_csv(file_path):
    """Reads a BVP signal from CSV (assumes 1D signal)."""
    df = pd.read_csv(file_path, header=None)
    if df.shape[0] == 1:
        return df.values.flatten()
    elif df.shape[1] == 1:
        return df.values[:, 0]
    else:
        raise ValueError("CSV file should contain a single BVP signal in one row or column.")

def estimate_hr_fft(bvp_signal, fs=30, plot_spectrum=False):
    """Estimates heart rate from BVP signal using FFT."""
    N = len(bvp_signal)
    bvp_detrended = detrend(bvp_signal)
    
    freqs = np.fft.fftfreq(N, d=1/fs)
    fft_spectrum = np.abs(fft(bvp_detrended))**2

    # Only keep positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_spectrum = fft_spectrum[pos_mask]

    # Limit to physiological HR range [0.7–3 Hz] = [42–180 bpm]
    hr_mask = (freqs >= 0.7) & (freqs <= 3.0)
    freqs_hr = freqs[hr_mask]
    spectrum_hr = fft_spectrum[hr_mask]

    if len(freqs_hr) == 0:
        return None  # No valid frequency band

    peak_idx = np.argmax(spectrum_hr)
    peak_freq = freqs_hr[peak_idx]
    estimated_hr_bpm = peak_freq * 60  # Hz to BPM

    if plot_spectrum:
        plt.plot(freqs_hr * 60, spectrum_hr)
        plt.xlabel('Frequency (BPM)')
        plt.ylabel('Power')
        plt.title('Heart Rate Spectrum')
        plt.grid(True)
        plt.show()

    return estimated_hr_bpm

# %%

# Example usage
file_path = "/hpc/group/dunnlab/rppg_data/data/DATASET_1_IN/s2/bvp_s2_T3.csv"  # replace with your file
bvp_signal = read_bvp_from_csv(file_path)
estimated_hr = estimate_hr_fft(bvp_signal, fs=30, plot_spectrum=True)
print(f"Estimated Heart Rate: {estimated_hr:.2f} BPM")

# %%
import os
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def read_bvp_from_csv(file_path):
    """Reads a BVP signal from CSV (assumes 1D signal)."""
    df = pd.read_csv(file_path, header=None)
    if df.shape[0] == 1:
        return df.values.flatten()
    elif df.shape[1] == 1:
        return df.values[:, 0]
    else:
        raise ValueError("CSV file should contain a single BVP signal in one row or column.")

def estimate_hr_fft(bvp_signal, fs=30):
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
    hr_mask = (freqs >= 0.7) & (freqs <= 3.0)
    freqs_hr = freqs[hr_mask]
    spectrum_hr = fft_spectrum[hr_mask]

    if len(freqs_hr) == 0:
        return None, None, None

    peak_idx = np.argmax(spectrum_hr)
    peak_freq = freqs_hr[peak_idx]
    estimated_hr_bpm = peak_freq * 60

    return estimated_hr_bpm, freqs_hr * 60, spectrum_hr  # HR in BPM

def plot_hr_spectrum(freqs_bpm, spectrum, file_name, output_dir):
    """Plots and saves HR spectrum for a given BVP signal."""
    plt.figure()
    plt.plot(freqs_bpm, spectrum)
    plt.xlabel('Frequency (BPM)')
    plt.ylabel('Power')
    plt.title(f'HR Spectrum: {file_name}')
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{file_name}.png")
    print(save_path)
    plt.savefig(save_path)
    plt.close()

def process_all_bvp_in_dir(root_dir, fs=30, output_dir="/hpc/group/dunnlab/rppg_data/data/meta_d_1"):
    """Walk through all CSVs under root_dir and generate HR plots."""
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".csv") and "bvp" in file.lower():
                full_path = os.path.join(subdir, file)
                try:
                    bvp = read_bvp_from_csv(full_path)
                    hr_bpm, freqs_bpm, spectrum = estimate_hr_fft(bvp, fs=fs)
                    if hr_bpm is not None:
                        print(f"{file}: {hr_bpm:.2f} BPM")
                        file_id = os.path.splitext(file)[0]
                        plot_hr_spectrum(freqs_bpm, spectrum, file_id, output_dir)
                    else:
                        print(f"{file}: No valid HR frequency detected.")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

# Usage
process_all_bvp_in_dir("/hpc/group/dunnlab/rppg_data/data/DATASET_1_IN", fs=30)

# %%
