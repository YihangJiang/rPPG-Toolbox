# %%
"""
Plot PPG signals (Label and Pred) from a chunk CSV and estimate heart rate for both.
Run cells in order. Edit csv_path in the config cell to analyze different chunks.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import butter

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evaluation.post_process import _calculate_fft_hr, _detrend

# %%
# Config: set path to PPG_signals.csv
csv_path = _project_root / (
    "scripts/test_runs/exp/PURE-IN_ClipLength240_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_"
    "Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_SizeW96_SizeH96/"
    "ppg_plots/103/chunk_0_PPG_signals.csv"
)
fs = 30  # Hz
diff_flag = True  # True if signals are DiffNormalized (derivative)

# %%
def estimate_hr_from_ppg(signal, fs=30, diff_flag=True, use_bandpass=True):
    """Estimate heart rate from PPG signal using FFT."""
    low_pass, high_pass = 0.75, 2.5
    signal = np.asarray(signal, dtype=np.float64).flatten()
    if diff_flag:
        signal = np.cumsum(signal)
    signal = _detrend(signal, 100)
    if use_bandpass:
        [b, a] = butter(1, [low_pass / fs * 2, high_pass / fs * 2], btype='bandpass')
        signal = scipy.signal.filtfilt(b, a, signal)
    hr_bpm = _calculate_fft_hr(signal, fs=fs, low_pass=low_pass, high_pass=high_pass)
    return hr_bpm

# %%
# Load PPG signals
df = pd.read_csv(csv_path)
label = df['Label'].values.astype(np.float64)
pred = df['Pred'].values.astype(np.float64)

# %%
# Estimate heart rate for both
hr_label = estimate_hr_from_ppg(label, fs=fs, diff_flag=diff_flag)
hr_pred = estimate_hr_from_ppg(pred, fs=fs, diff_flag=diff_flag)
print(f"Label HR (FFT): {hr_label:.1f} bpm")
print(f"Pred  HR (FFT): {hr_pred:.1f} bpm")
print(f"HR error: {abs(hr_pred - hr_label):.1f} bpm")

# %%
# Plot Label and Pred
time_axis = np.arange(len(label)) / fs
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(time_axis, label, label=f'Label (GT) — HR: {hr_label:.1f} bpm', alpha=0.8, linewidth=1.2)
ax.plot(time_axis, pred, label=f'Pred — HR: {hr_pred:.1f} bpm', alpha=0.8, linewidth=1.2)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('PPG Signal', fontsize=12)
ax.set_title(f'PPG Comparison — {csv_path.parent.name}/{csv_path.name}', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# %%
# Save plot
output_path = csv_path.with_suffix('.pdf')
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Plot saved to: {output_path.resolve()}")

# %%
