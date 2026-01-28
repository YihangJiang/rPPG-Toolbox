# %%
"""

Recalculate FFT metrics (MAE, RMSE, MAPE, Pearson, SNR) from saved CSV files.

This script reads the per-chunk metrics CSV file and recalculates the overall
metrics that were printed during testing. The calculations exactly match
the implementation in evaluation/metrics.py to ensure consistency.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def recalculate_metrics_from_csv(csv_path):
    """
    Recalculate FFT metrics from per-chunk metrics CSV file.
    
    Args:
        csv_path: Path to the per_chunk_metrics.csv file
        
    Returns:
        Dictionary with calculated metrics
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract ground truth and predicted HR values
    gt_hr_fft_all = df['gt_hr'].values
    pred_hr_fft_all = df['pred_hr'].values
    
    # Extract SNR if available
    snr_all = None
    if 'SNR' in df.columns:
        snr_all = df['SNR'].values
    
    num_test_samples = len(pred_hr_fft_all)
    
    print(f"Loaded {num_test_samples} chunks from: {csv_path}")
    print("=" * 80)
    
    # Calculate metrics (matching the code in evaluation/metrics.py)
    metrics = {}
    
    # MAE
    mae = np.mean(np.abs(pred_hr_fft_all - gt_hr_fft_all))
    mae_std = np.std(np.abs(pred_hr_fft_all - gt_hr_fft_all))
    mae_se = mae_std / np.sqrt(num_test_samples)
    metrics['MAE'] = {'value': mae, 'std': mae_std, 'se': mae_se}
    print(f"FFT MAE (FFT Label): {mae} +/- {mae_se}")
    
    # RMSE
    squared_errors = np.square(pred_hr_fft_all - gt_hr_fft_all)
    rmse = np.sqrt(np.mean(squared_errors))
    rmse_std = np.std(squared_errors)
    rmse_se = np.sqrt(rmse_std / np.sqrt(num_test_samples))
    metrics['RMSE'] = {'value': rmse, 'std': rmse_std, 'se': rmse_se}
    print(f"FFT RMSE (FFT Label): {rmse} +/- {rmse_se}")
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((pred_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
    mape_std = np.std(np.abs((pred_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all))
    mape_se = mape_std / np.sqrt(num_test_samples) * 100
    metrics['MAPE'] = {'value': mape, 'std': mape_std, 'se': mape_se}
    print(f"FFT MAPE (FFT Label): {mape} +/- {mape_se}")
    
    # Pearson Correlation
    pearson_matrix = np.corrcoef(pred_hr_fft_all, gt_hr_fft_all)
    pearson = pearson_matrix[0][1]
    pearson_se = np.sqrt((1 - pearson**2) / (num_test_samples - 2))
    metrics['Pearson'] = {'value': pearson, 'se': pearson_se}
    print(f"FFT Pearson (FFT Label): {pearson} +/- {pearson_se}")
    
    # SNR (Signal-to-Noise Ratio) - if available in CSV
    if snr_all is not None:
        snr_mean = np.mean(snr_all)
        snr_se = np.std(snr_all) / np.sqrt(num_test_samples)
        metrics['SNR'] = {'value': snr_mean, 'std': np.std(snr_all), 'se': snr_se}
        print(f"FFT SNR (FFT Label): {snr_mean} +/- {snr_se} (dB)")
    else:
        print("FFT SNR (FFT Label): Not available (SNR column not found in CSV)")
    
    # Additional statistics
    print("=" * 80)
    print(f"Number of chunks: {num_test_samples}")
    print(f"Number of videos: {df['video_id'].nunique()}")
    print(f"GT HR range: [{gt_hr_fft_all.min():.2f}, {gt_hr_fft_all.max():.2f}] bpm")
    print(f"Pred HR range: [{pred_hr_fft_all.min():.2f}, {pred_hr_fft_all.max():.2f}] bpm")
    print(f"GT HR mean: {gt_hr_fft_all.mean():.2f} +/- {gt_hr_fft_all.std():.2f} bpm")
    print(f"Pred HR mean: {pred_hr_fft_all.mean():.2f} +/- {pred_hr_fft_all.std():.2f} bpm")
    
    return metrics


# Hardcoded CSV path - modify this to point to your per_chunk_metrics.csv file
script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd() / 'analysis'
project_root = script_dir.parent
csv_path = project_root / "scripts" / "test_runs" / "exp" / \
           "UBFC-PHYS-IN_ClipLength240_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_SizeW96_SizeH96_g" / \
           "saved_test_outputs" / "deepphys_ubfc_rppg_to_phys_in_per_chunk_metrics.csv"

# Run the calculation
if Path(csv_path).exists():
    print(f"Recalculating metrics from: {csv_path}")
    print("=" * 80)
    metrics = recalculate_metrics_from_csv(csv_path)
    print("\n" + "=" * 80)
    print("Metrics recalculation complete!")
else:
    print(f"Error: CSV file not found at: {csv_path}")
    print("Please update the csv_path variable in the script.")


# %%
