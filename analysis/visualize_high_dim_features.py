# %%
"""
High-dimensional feature visualization for rPPG model performance analysis.

This script extracts and visualizes features from:
1. Raw signal labels (PPG/BVP signals) - frequency domain, statistical, temporal features
2. Raw video data - optical flow, color statistics, motion, texture features

Features are then visualized using dimensionality reduction (PCA, t-SNE, UMAP)
and correlated with model performance metrics to identify common patterns
in poorly performing samples.
"""

import pickle
import pandas as pd
import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import umap, but make it optional
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available. Install with: pip install umap-learn")

def extract_signal_features(signal_data, fs=30):
    """Extract high-dimensional features from PPG/BVP signal.
    
    Args:
        signal_data: 1D array of signal values
        fs: Sampling frequency
        
    Returns:
        dict: Dictionary of extracted features
    """
    signal_data = np.array(signal_data).flatten()
    features = {}
    
    # Statistical features
    features['mean'] = np.mean(signal_data)
    features['std'] = np.std(signal_data)
    features['var'] = np.var(signal_data)
    features['skewness'] = stats.skew(signal_data)
    features['kurtosis'] = stats.kurtosis(signal_data)
    features['min'] = np.min(signal_data)
    features['max'] = np.max(signal_data)
    features['range'] = features['max'] - features['min']
    features['median'] = np.median(signal_data)
    features['q25'] = np.percentile(signal_data, 25)
    features['q75'] = np.percentile(signal_data, 75)
    features['iqr'] = features['q75'] - features['q25']
    
    # Temporal features
    features['zero_crossings'] = len(np.where(np.diff(np.signbit(signal_data)))[0])
    features['autocorr_lag1'] = np.corrcoef(signal_data[:-1], signal_data[1:])[0, 1] if len(signal_data) > 1 else 0
    
    # Frequency domain features
    if len(signal_data) > 1:
        # FFT
        fft_vals = np.abs(fft(signal_data))
        freqs = fftfreq(len(signal_data), 1/fs)
        
        # Only use positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_vals[:len(fft_vals)//2]
        
        # Heart rate range (0.5-4 Hz = 30-240 BPM)
        hr_mask = (pos_freqs >= 0.5) & (pos_freqs <= 4.0)
        if np.any(hr_mask):
            features['dominant_freq'] = pos_freqs[np.argmax(pos_fft[hr_mask])]
            features['spectral_centroid'] = np.sum(pos_freqs * pos_fft) / (np.sum(pos_fft) + 1e-10)
            features['spectral_rolloff'] = np.sum(pos_freqs[pos_fft > 0.8 * np.max(pos_fft)])
            features['spectral_bandwidth'] = np.sqrt(np.sum(((pos_freqs - features['spectral_centroid'])**2) * pos_fft) / (np.sum(pos_fft) + 1e-10))
            features['spectral_flatness'] = np.exp(np.mean(np.log(pos_fft + 1e-10))) / (np.mean(pos_fft) + 1e-10)
        else:
            features['dominant_freq'] = 0
            features['spectral_centroid'] = 0
            features['spectral_rolloff'] = 0
            features['spectral_bandwidth'] = 0
            features['spectral_flatness'] = 0
        
        # Power in different frequency bands
        very_low = (pos_freqs >= 0.04) & (pos_freqs < 0.15)  # Very low frequency
        low = (pos_freqs >= 0.15) & (pos_freqs < 0.4)  # Low frequency
        high = (pos_freqs >= 0.4) & (pos_freqs < 0.6)  # High frequency
        hr_band = (pos_freqs >= 0.5) & (pos_freqs <= 4.0)  # Heart rate band
        
        features['power_vlf'] = np.sum(pos_fft[very_low]) if np.any(very_low) else 0
        features['power_lf'] = np.sum(pos_fft[low]) if np.any(low) else 0
        features['power_hf'] = np.sum(pos_fft[high]) if np.any(high) else 0
        features['power_hr'] = np.sum(pos_fft[hr_band]) if np.any(hr_band) else 0
        features['lf_hf_ratio'] = features['power_lf'] / (features['power_hf'] + 1e-10)
    else:
        # Default values if signal is too short
        for key in ['dominant_freq', 'spectral_centroid', 'spectral_rolloff', 
                   'spectral_bandwidth', 'spectral_flatness', 'power_vlf', 
                   'power_lf', 'power_hf', 'power_hr', 'lf_hf_ratio']:
            features[key] = 0
    
    # Derivative features
    if len(signal_data) > 1:
        diff_signal = np.diff(signal_data)
        features['diff_mean'] = np.mean(np.abs(diff_signal))
        features['diff_std'] = np.std(diff_signal)
        features['diff_max'] = np.max(np.abs(diff_signal))
    else:
        features['diff_mean'] = 0
        features['diff_std'] = 0
        features['diff_max'] = 0
    
    return features

def extract_video_features(video_chunk):
    """Extract high-dimensional features from video chunk.
    
    Args:
        video_chunk: Video array of shape (T, H, W, C) or (T, C, H, W)
        
    Returns:
        dict: Dictionary of extracted features
    """
    # Handle different input formats
    if len(video_chunk.shape) == 4:
        if video_chunk.shape[-1] == 3 or video_chunk.shape[-1] == 1:  # (T, H, W, C)
            T, H, W, C = video_chunk.shape
            video_chunk = np.transpose(video_chunk, (0, 3, 1, 2))  # Convert to (T, C, H, W)
        else:  # Already (T, C, H, W)
            T, C, H, W = video_chunk.shape
    else:
        raise ValueError(f"Unexpected video shape: {video_chunk.shape}")
    
    # Convert to (T, H, W, C) for easier processing
    video_chunk = np.transpose(video_chunk, (0, 2, 3, 1))
    T, H, W, C = video_chunk.shape
    
    features = {}
    
    # Color statistics per channel
    for ch_idx in range(C):
        ch_data = video_chunk[:, :, :, ch_idx]
        features[f'ch{ch_idx}_mean'] = np.mean(ch_data)
        features[f'ch{ch_idx}_std'] = np.std(ch_data)
        features[f'ch{ch_idx}_min'] = np.min(ch_data)
        features[f'ch{ch_idx}_max'] = np.max(ch_data)
        features[f'ch{ch_idx}_median'] = np.median(ch_data)
    
    # Overall brightness and contrast
    if C >= 3:  # RGB
        gray = np.mean(video_chunk[:, :, :, :3], axis=3)  # Convert to grayscale
        features['brightness_mean'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['contrast'] = np.std(gray)
    else:
        gray = video_chunk[:, :, :, 0]
        features['brightness_mean'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['contrast'] = np.std(gray)
    
    # Temporal variation (motion)
    if T > 1:
        frame_diffs = np.diff(gray, axis=0)
        features['motion_mean'] = np.mean(np.abs(frame_diffs))
        features['motion_std'] = np.std(frame_diffs)
        features['motion_max'] = np.max(np.abs(frame_diffs))
        
        # Optical flow approximation (simplified)
        flow_magnitude = []
        for t in range(T - 1):
            frame1 = gray[t].astype(np.float32)
            frame2 = gray[t + 1].astype(np.float32)
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            flow_magnitude.append(np.mean(flow_mag))
        
        if flow_magnitude:
            features['optical_flow_mean'] = np.mean(flow_magnitude)
            features['optical_flow_std'] = np.std(flow_magnitude)
            features['optical_flow_max'] = np.max(flow_magnitude)
        else:
            features['optical_flow_mean'] = 0
            features['optical_flow_std'] = 0
            features['optical_flow_max'] = 0
    else:
        features['motion_mean'] = 0
        features['motion_std'] = 0
        features['motion_max'] = 0
        features['optical_flow_mean'] = 0
        features['optical_flow_std'] = 0
        features['optical_flow_max'] = 0
    
    # Texture features (using Laplacian variance as texture measure)
    texture_scores = []
    for t in range(T):
        frame = gray[t].astype(np.uint8)
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        texture_scores.append(np.var(laplacian))
    
    features['texture_mean'] = np.mean(texture_scores)
    features['texture_std'] = np.std(texture_scores)
    features['texture_max'] = np.max(texture_scores)
    
    # Spatial gradient features
    gradient_magnitudes = []
    for t in range(T):
        frame = gray[t].astype(np.uint8)
        grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitudes.append(np.mean(grad_mag))
    
    features['gradient_mean'] = np.mean(gradient_magnitudes)
    features['gradient_std'] = np.std(gradient_magnitudes)
    
    # Color distribution (histogram statistics)
    if C >= 3:
        for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
            ch_data = video_chunk[:, :, :, ch_idx].flatten()
            hist, _ = np.histogram(ch_data, bins=32, range=(0, 255))
            hist = hist / (np.sum(hist) + 1e-10)
            features[f'{ch_name}_hist_entropy'] = -np.sum(hist * np.log(hist + 1e-10))
    
    return features

def load_test_data(test_results_dir):
    """Load test results from pickle and CSV files.
    
    The pickle file contains:
    - 'predictions': dict[video_id][chunk_index] -> prediction array (model outputs)
    - 'labels': dict[video_id][chunk_index] -> label array (ground truth PPG/BVP signals)
    - 'label_type': str (e.g., 'DiffNormalized')
    - 'fs': int (sampling frequency in Hz)
    
    The pickle file is saved during model testing at:
    {test_results_dir}/saved_test_outputs/{model_file_name}_outputs.pickle
    
    Args:
        test_results_dir: Directory containing test results (should contain 'saved_test_outputs' subdirectory)
        
    Returns:
        tuple: (predictions_dict, labels_dict, metrics_df, metadata)
    """
    saved_outputs_dir = os.path.join(test_results_dir, 'saved_test_outputs')
    
    if not os.path.exists(saved_outputs_dir):
        raise FileNotFoundError(
            f"Directory not found: {saved_outputs_dir}\n"
            f"The pickle file should be in: {saved_outputs_dir}\n"
            f"Expected structure: {test_results_dir}/saved_test_outputs/*_outputs.pickle\n"
            f"This file is created automatically during model testing."
        )
    
    # Find pickle file
    pickle_files = [f for f in os.listdir(saved_outputs_dir) if f.endswith('.pickle')]
    if not pickle_files:
        available_files = os.listdir(saved_outputs_dir)
        raise FileNotFoundError(
            f"No pickle file found in {saved_outputs_dir}\n"
            f"Available files: {available_files}\n"
            f"Expected file pattern: *_outputs.pickle\n"
            f"The pickle file is created during testing when config.TEST.OUTPUT_SAVE_DIR is set."
        )
    pickle_path = os.path.join(saved_outputs_dir, pickle_files[0])
    
    # Load pickle
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    predictions = data['predictions']
    labels = data['labels']
    fs = data.get('fs', 30)
    label_type = data.get('label_type', 'DiffNormalized')
    
    # Load metrics CSV (may not contain all chunks, only those with metrics calculated)
    metrics_files = [f for f in os.listdir(saved_outputs_dir) if 'per_chunk' in f and f.endswith('.csv')]
    if not metrics_files:
        print(f"Warning: No per_chunk metrics CSV found in {saved_outputs_dir}")
        metrics_df = None
    else:
        metrics_path = os.path.join(saved_outputs_dir, metrics_files[0])
        metrics_df = pd.read_csv(metrics_path)
    
    return predictions, labels, metrics_df, {'fs': fs, 'label_type': label_type}

def find_video_cache_file(cached_path, video_id, chunk_index, file_list_cache=None):
    """Find the cached video file for a given video_id and chunk_index.
    
    Args:
        cached_path: Path to cached preprocessed data (base path, may have subdirectories)
        video_id: Video/subject ID
        chunk_index: Chunk index within the video
        file_list_cache: Optional dict mapping (video_id, chunk_index) to file paths
        
    Returns:
        str or None: Path to the cached video file, or None if not found
    """
    # First check cache if provided
    if file_list_cache is not None:
        key = (str(video_id), int(chunk_index))
        if key in file_list_cache:
            file_path = file_list_cache[key]
            if os.path.exists(file_path):
                return file_path
    
    # Try different naming patterns
    patterns = [
        f"{video_id}_input{chunk_index}.npy",
        f"subject{video_id}_input{chunk_index}.npy",
        f"{int(video_id):04d}_input{chunk_index}.npy",
    ]
    
    # Try in base cached_path
    for pattern in patterns:
        file_path = os.path.join(cached_path, pattern)
        if os.path.exists(file_path):
            return file_path
    
    # Search in subdirectories (EXP_DATA_NAME folders)
    if os.path.exists(cached_path):
        for item in os.listdir(cached_path):
            item_path = os.path.join(cached_path, item)
            if os.path.isdir(item_path):
                for pattern in patterns:
                    file_path = os.path.join(item_path, pattern)
                    if os.path.exists(file_path):
                        return file_path
    
    # If exact match not found, search in file list
    file_list_dir = os.path.join(cached_path, 'DataFileLists')
    if not os.path.exists(file_list_dir):
        # Try in subdirectories
        if os.path.exists(cached_path):
            for item in os.listdir(cached_path):
                item_path = os.path.join(cached_path, item)
                if os.path.isdir(item_path):
                    test_file_list_dir = os.path.join(item_path, 'DataFileLists')
                    if os.path.exists(test_file_list_dir):
                        file_list_dir = test_file_list_dir
                        break
    
    if os.path.exists(file_list_dir):
        for file_list in os.listdir(file_list_dir):
            if file_list.endswith('.csv'):
                file_list_path = os.path.join(file_list_dir, file_list)
                try:
                    df = pd.read_csv(file_list_path)
                    for input_file in df['input_files']:
                        filename = os.path.basename(input_file)
                        # Check if this file matches our video_id and chunk_index
                        if f"{video_id}" in filename and f"input{chunk_index}" in filename:
                            # Check if file exists
                            if os.path.exists(input_file):
                                return input_file
                            # Try relative to cached_path
                            rel_path = os.path.join(cached_path, os.path.basename(input_file))
                            if os.path.exists(rel_path):
                                return rel_path
                            # Try in subdirectories
                            if os.path.exists(cached_path):
                                for item in os.listdir(cached_path):
                                    item_path = os.path.join(cached_path, item)
                                    if os.path.isdir(item_path):
                                        test_path = os.path.join(item_path, os.path.basename(input_file))
                                        if os.path.exists(test_path):
                                            return test_path
                except Exception as e:
                    continue
    
    return None

def build_file_list_cache(cached_path):
    """Build a cache mapping (video_id, chunk_index) to file paths.
    
    Args:
        cached_path: Path to cached preprocessed data
        
    Returns:
        dict: Mapping (video_id, chunk_index) -> file_path
    """
    cache = {}
    
    # Search in file lists
    file_list_dir = os.path.join(cached_path, 'DataFileLists')
    if not os.path.exists(file_list_dir):
        # Try in subdirectories
        if os.path.exists(cached_path):
            for item in os.listdir(cached_path):
                item_path = os.path.join(cached_path, item)
                if os.path.isdir(item_path):
                    test_file_list_dir = os.path.join(item_path, 'DataFileLists')
                    if os.path.exists(test_file_list_dir):
                        file_list_dir = test_file_list_dir
                        break
    
    if os.path.exists(file_list_dir):
        for file_list in os.listdir(file_list_dir):
            if file_list.endswith('.csv'):
                file_list_path = os.path.join(file_list_dir, file_list)
                try:
                    df = pd.read_csv(file_list_path)
                    for input_file in df['input_files']:
                        if os.path.exists(input_file):
                            filename = os.path.basename(input_file)
                            # Parse video_id and chunk_index from filename
                            # Format: {video_id}_input{chunk_index}.npy
                            if '_input' in filename:
                                parts = filename.replace('.npy', '').split('_input')
                                if len(parts) == 2:
                                    video_id = parts[0]
                                    try:
                                        chunk_index = int(parts[1])
                                        cache[(str(video_id), chunk_index)] = input_file
                                    except ValueError:
                                        continue
                except Exception as e:
                    continue
    
    return cache

# %%
# Configuration
# Expand ~ to home directory and convert to absolute path
test_results_dir = os.path.expanduser("~/Desktop/rPPG-Toolbox/scripts/test_runs/exp/PURE_ClipLength240_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_SizeW72_SizeH72")
test_results_dir = os.path.abspath(test_results_dir)

# Try to infer cached path from config or use default
# For PURE dataset, default cached path
cached_path = "/mnt/nvme2/rppg_data/DATASET_PRE"

output_dir = os.path.join(test_results_dir, "feature_analysis")
os.makedirs(output_dir, exist_ok=True)

print("Loading test data...")
predictions, labels, metrics_df, metadata = load_test_data(test_results_dir)
fs = metadata['fs']

if metrics_df is not None:
    print(f"Found {len(metrics_df)} chunks in metrics CSV")
print(f"Sampling frequency: {fs} Hz")

# Build file list cache for faster lookup
print("\nBuilding file list cache...")
file_list_cache = build_file_list_cache(cached_path)
print(f"Found {len(file_list_cache)} cached files in file lists")

# Extract features for each chunk
print("\nExtracting features...")
signal_features_list = []
video_features_list = []
chunk_info = []

# Build a lookup for metrics by (video_id, chunk_index)
metrics_lookup = {}
if metrics_df is not None:
    print(f"\nBuilding metrics lookup from CSV...")
    print(f"  CSV columns: {list(metrics_df.columns)}")
    print(f"  CSV shape: {metrics_df.shape}")
    
    # Check for required columns
    required_cols = ['video_id', 'chunk_index']
    missing_cols = [col for col in required_cols if col not in metrics_df.columns]
    if missing_cols:
        print(f"  WARNING: Missing required columns: {missing_cols}")
        print(f"  Available columns: {list(metrics_df.columns)}")
    
    def normalize_video_id(video_id):
        """Normalize video_id to match pickle format.
        
        Handles cases where CSV has '1001.0' but pickle has '1001'
        """
        # Convert to string and remove trailing '.0' if present
        vid_str = str(video_id)
        # Remove trailing .0 if it's a float representation
        if vid_str.endswith('.0'):
            vid_str = vid_str[:-2]
        return vid_str
    
    for idx, row in metrics_df.iterrows():
        try:
            # Normalize video_id to handle float->string conversion (e.g., '1001.0' -> '1001')
            video_id_normalized = normalize_video_id(row['video_id'])
            key = (video_id_normalized, int(row['chunk_index']))
            metrics_lookup[key] = {
                'SNR': row.get('SNR', np.nan),
                'MACC': row.get('MACC', np.nan),
                'gt_hr': row.get('gt_hr', np.nan),
                'pred_hr': row.get('pred_hr', np.nan),
            }
        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            print(f"    Row data: {row.to_dict()}")
    
    print(f"  Built lookup with {len(metrics_lookup)} entries")
    if len(metrics_lookup) > 0:
        sample_keys = list(metrics_lookup.keys())[:5]
        print(f"  Sample keys (after normalization): {sample_keys}")
else:
    print(f"\nWARNING: metrics_df is None - no metrics CSV found!")

# Process all chunks from the pickle file (not just those in metrics CSV)
print(f"\nProcessing all chunks from pickle file...")
total_chunks = sum(len(chunks) for chunks in labels.values())
print(f"Total chunks to process: {total_chunks}")

# Debug: Print sample video_id and chunk_idx from pickle
sample_video_ids = list(labels.keys())[:3]
print(f"Sample video_ids from pickle: {sample_video_ids}")
for vid_id in sample_video_ids:
    sample_chunks = list(labels[vid_id].keys())[:3]
    print(f"  Video {vid_id}: chunk indices {sample_chunks} (total: {len(labels[vid_id])} chunks)")

# Limit to first N chunks for faster processing (remove limit for full analysis)
max_chunks = None  # Set to None for all chunks, or a number like 100 for testing
chunk_count = 0

# Track metrics matching
metrics_found_count = 0
metrics_not_found_count = 0
sample_not_found_keys = []

with tqdm(total=min(max_chunks, total_chunks) if max_chunks else total_chunks, desc="Processing chunks") as pbar:
    for video_id in labels.keys():
        for chunk_idx in labels[video_id].keys():
            if max_chunks and chunk_count >= max_chunks:
                break
            
            video_id_str = str(video_id)
            chunk_idx_int = int(chunk_idx)
            
            # Get signal data
            signal_data = labels[video_id][chunk_idx].numpy() if hasattr(labels[video_id][chunk_idx], 'numpy') else np.array(labels[video_id][chunk_idx])
            
            # Extract signal features
            sig_feat = extract_signal_features(signal_data, fs=fs)
            signal_features_list.append(sig_feat)
            
            # Try to load video chunk
            video_file = find_video_cache_file(cached_path, video_id_str, chunk_idx_int, file_list_cache)
            if video_file and os.path.exists(video_file):
                try:
                    video_chunk = np.load(video_file)
                    vid_feat = extract_video_features(video_chunk)
                    video_features_list.append(vid_feat)
                except Exception as e:
                    print(f"Warning: Could not load video for {video_id_str}_chunk{chunk_idx_int}: {e}")
                    video_features_list.append({})
            else:
                video_features_list.append({})
            
            # Get metrics from lookup (if available)
            key = (video_id_str, chunk_idx_int)
            if key in metrics_lookup:
                metrics = metrics_lookup[key]
                metrics_found_count += 1
            else:
                # If metrics not found, use NaN
                metrics = {
                    'SNR': np.nan,
                    'MACC': np.nan,
                    'gt_hr': np.nan,
                    'pred_hr': np.nan,
                }
                metrics_not_found_count += 1
                if len(sample_not_found_keys) < 5:
                    sample_not_found_keys.append(key)
            
            # Calculate HR difference (predicted - ground truth)
            # Positive = overestimation, Negative = underestimation
            gt_hr_val = metrics['gt_hr']
            pred_hr_val = metrics['pred_hr']
            if not (np.isnan(gt_hr_val) or np.isnan(pred_hr_val)):
                hr_diff = pred_hr_val - gt_hr_val
                hr_abs_diff = np.abs(hr_diff)
            else:
                hr_diff = np.nan
                hr_abs_diff = np.nan
            
            # Store chunk info
            chunk_info.append({
                'video_id': video_id_str,
                'chunk_index': chunk_idx_int,
                'SNR': metrics['SNR'],
                'MACC': metrics['MACC'],
                'gt_hr': metrics['gt_hr'],
                'pred_hr': metrics['pred_hr'],
                'hr_diff': hr_diff,  # pred_hr - gt_hr (signed difference)
                'hr_abs_diff': hr_abs_diff,  # |pred_hr - gt_hr| (absolute error)
            })
            
            chunk_count += 1
            pbar.update(1)
        
        if max_chunks and chunk_count >= max_chunks:
            break

# Print metrics matching summary
print(f"\n{'='*60}")
print(f"Metrics Matching Summary:")
print(f"  Metrics found: {metrics_found_count}/{chunk_count} chunks ({100*metrics_found_count/max(chunk_count,1):.1f}%)")
print(f"  Metrics NOT found: {metrics_not_found_count}/{chunk_count} chunks ({100*metrics_not_found_count/max(chunk_count,1):.1f}%)")
if sample_not_found_keys:
    print(f"  Sample keys NOT found in metrics CSV: {sample_not_found_keys}")
    if len(metrics_lookup) > 0:
        sample_found_keys = list(metrics_lookup.keys())[:3]
        print(f"  Sample keys FOUND in metrics CSV: {sample_found_keys}")
        print(f"  NOTE: Check if video_id/chunk_index formats match between pickle and CSV!")
print(f"{'='*60}")

# Convert to DataFrames
print("\nConverting to DataFrames...")
print(f"  Signal features extracted: {len(signal_features_list)} chunks")
print(f"  Video features extracted: {len(video_features_list)} chunks")
print(f"  Chunk info: {len(chunk_info)} chunks")

# Check if we have any features
if len(signal_features_list) == 0:
    raise ValueError("No signal features extracted! Check that labels are loaded correctly from pickle file.")

# Count non-empty feature dicts
non_empty_signal = sum(1 for f in signal_features_list if len(f) > 0)
non_empty_video = sum(1 for f in video_features_list if len(f) > 0)
print(f"  Non-empty signal feature dicts: {non_empty_signal}/{len(signal_features_list)}")
print(f"  Non-empty video feature dicts: {non_empty_video}/{len(video_features_list)}")

signal_df = pd.DataFrame(signal_features_list)
video_df = pd.DataFrame(video_features_list)
info_df = pd.DataFrame(chunk_info)

print(f"  Signal DF shape: {signal_df.shape}, columns: {list(signal_df.columns)[:5]}...")
print(f"  Video DF shape: {video_df.shape}, columns: {list(video_df.columns)[:5] if len(video_df.columns) > 0 else 'None'}...")
print(f"  Info DF shape: {info_df.shape}, columns: {list(info_df.columns)}")

# Check for NaN values in info_df
metric_cols = ['SNR', 'MACC', 'gt_hr', 'pred_hr', 'hr_diff', 'hr_abs_diff']
for col in metric_cols:
    if col in info_df.columns:
        nan_count = info_df[col].isna().sum()
        total_count = len(info_df)
        if nan_count == total_count:
            print(f"  WARNING: Column '{col}' is ALL NaN ({nan_count}/{total_count})")
        elif nan_count > 0:
            print(f"  WARNING: Column '{col}' has {nan_count}/{total_count} NaN values")
        else:
            print(f"  OK: Column '{col}' has no NaN values")
# %%
# Combine all features - handle case where video features might be empty
if len(video_df.columns) > 0:
    feature_df = pd.concat([signal_df, video_df], axis=1)
else:
    print("  Warning: No video features found. Proceeding with signal features only.")
    feature_df = signal_df.copy()

feature_df = pd.concat([info_df, feature_df], axis=1)

print(f"  Combined feature DF shape: {feature_df.shape}")

# Remove rows with all NaN features
non_meta_cols = [col for col in feature_df.columns if col not in ['video_id', 'chunk_index', 'SNR', 'MACC', 'gt_hr', 'pred_hr', 'hr_diff', 'hr_abs_diff']]
feature_df = feature_df.dropna(how='all', subset=non_meta_cols)

print(f"\nExtracted features from {len(feature_df)} chunks")
print(f"Signal features: {len(signal_df.columns)}")
print(f"Video features: {len(video_df.columns)}")

# Check if we have any data
if len(feature_df) == 0:
    raise ValueError("No valid feature data extracted! Check that predictions and labels are loaded correctly.")

# Prepare features for dimensionality reduction
feature_cols = [col for col in feature_df.columns 
                if col not in ['video_id', 'chunk_index', 'SNR', 'MACC', 'gt_hr', 'pred_hr', 'hr_diff', 'hr_abs_diff']]

if len(feature_cols) == 0:
    raise ValueError("No feature columns found! Check that feature extraction is working correctly.")

print(f"Using {len(feature_cols)} feature columns for analysis")

# Save feature dataframe
feature_df.to_csv(os.path.join(output_dir, "extracted_features.csv"), index=False)
print(f"\nSaved features to {os.path.join(output_dir, 'extracted_features.csv')}")

# Extract feature matrix
X = feature_df[feature_cols].values

# Check for empty array
if X.shape[0] == 0 or X.shape[1] == 0:
    raise ValueError(f"Feature matrix is empty! Shape: {X.shape}. Check feature extraction.")

print(f"Feature matrix shape: {X.shape}")

# Handle NaN values - replace with column median
for col_idx in range(X.shape[1]):
    col_data = X[:, col_idx]
    nan_mask = np.isnan(col_data)
    if np.any(nan_mask):
        median_val = np.nanmedian(col_data)
        if np.isnan(median_val):
            median_val = 0.0
        X[nan_mask, col_idx] = median_val
        print(f"  Replaced {np.sum(nan_mask)} NaN values in column {feature_cols[col_idx]} with {median_val}")

# Handle inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Check for constant features (zero variance) and remove them
feature_variance = np.var(X, axis=0)
non_constant_mask = feature_variance > 1e-10
if np.sum(~non_constant_mask) > 0:
    removed_features = [feature_cols[i] for i in range(len(feature_cols)) if not non_constant_mask[i]]
    print(f"  Removing {len(removed_features)} constant features: {removed_features[:5]}...")
    X = X[:, non_constant_mask]
    feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if non_constant_mask[i]]

if X.shape[1] == 0:
    raise ValueError("All features are constant! Cannot perform dimensionality reduction.")

print(f"Final feature matrix shape: {X.shape}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute performance metric for coloring
# Options: 'SNR' (higher is better), 'hr_abs_diff' (lower is better), 'MACC' (higher is better)
performance_metric_name = 'SNR'  # Change to 'hr_abs_diff' or 'MACC' to use different metric

if performance_metric_name == 'hr_abs_diff':
    # For hr_abs_diff, invert so lower error = better (higher value in visualization)
    performance_metric = -feature_df['hr_abs_diff'].values  # Negative so lower error = better
    performance_metric_label = 'HR Absolute Error (inverted)'
elif performance_metric_name == 'MACC':
    performance_metric = feature_df['MACC'].values
    performance_metric_label = 'MACC'
else:  # Default: SNR
    performance_metric = feature_df['SNR'].values
    performance_metric_label = 'SNR'

performance_metric = np.nan_to_num(performance_metric, nan=0.0)

# Dimensionality reduction
print("\nPerforming dimensionality reduction...")

# PCA
print("  Computing PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# t-SNE
print("  Computing t-SNE (this may take a while)...")
# Perplexity must be less than n_samples
n_samples = len(X_scaled)
perplexity = min(30, max(5, n_samples - 1))  # At least 5, but less than n_samples
print(f"    Using perplexity={perplexity} for {n_samples} samples")
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP (if available)
if HAS_UMAP:
    print("  Computing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

# Visualization
print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2 if HAS_UMAP else 1, figsize=(16, 12) if HAS_UMAP else (16, 8))
if not HAS_UMAP:
    axes = [axes]
else:
    axes = axes.flatten()

# PCA plot
scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=performance_metric, 
                            cmap='RdYlGn', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
axes[0].set_title(f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0], label=performance_metric_label)

# t-SNE plot
scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=performance_metric,
                            cmap='RdYlGn', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
axes[1].set_title('t-SNE Visualization')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1], label=performance_metric_label)

# UMAP plot (if available)
if HAS_UMAP:
    scatter = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=performance_metric,
                                cmap='RdYlGn', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[2].set_title('UMAP Visualization')
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2], label=performance_metric_label)

# Feature correlation with performance
axes_idx = 3 if HAS_UMAP else 1
if HAS_UMAP:
    axes[axes_idx].axis('off')
    axes_idx = 3
else:
    axes_idx = 1

# Top correlated features
correlations = {}
for col in feature_cols:
    if col in feature_df.columns:
        corr = np.corrcoef(feature_df[col].fillna(0), performance_metric)[0, 1]
        if not np.isnan(corr):
            correlations[col] = corr

top_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
top_feat_names = [f[0] for f in top_features]
top_feat_corrs = [f[1] for f in top_features]

y_pos = np.arange(len(top_feat_names))
colors = ['red' if c < 0 else 'green' for c in top_feat_corrs]
bars = axes[axes_idx].barh(y_pos, top_feat_corrs, color=colors, alpha=0.7)
axes[axes_idx].set_yticks(y_pos)
axes[axes_idx].set_yticklabels(top_feat_names, fontsize=10, ha='right')
axes[axes_idx].set_xlabel(f'Correlation with {performance_metric_label}', fontsize=11)
axes[axes_idx].set_title(f'Top 15 Features Correlated with Performance ({performance_metric_label})', fontsize=12, fontweight='bold')
axes[axes_idx].grid(True, alpha=0.3, axis='x')
axes[axes_idx].axvline(x=0, color='black', linestyle='--', linewidth=0.5)

# Add value labels on bars
for i, (bar, corr_val) in enumerate(zip(bars, top_feat_corrs)):
    width = bar.get_width()
    label_x = width + (0.02 if width >= 0 else -0.02)
    axes[axes_idx].text(label_x, bar.get_y() + bar.get_height()/2, 
                       f'{corr_val:.3f}', 
                       ha='left' if width >= 0 else 'right', 
                       va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_visualization.png"), dpi=300, bbox_inches='tight')
print(f"Saved visualization to {os.path.join(output_dir, 'feature_visualization.png')}")

# Save correlation analysis
corr_col_name = f'correlation_with_{performance_metric_name}'
corr_df = pd.DataFrame(list(correlations.items()), columns=['feature', corr_col_name])
corr_df = corr_df.sort_values(corr_col_name, key=abs, ascending=False)
corr_filename = f"feature_correlations_with_{performance_metric_name}.csv"
corr_df.to_csv(os.path.join(output_dir, corr_filename), index=False)
print(f"Saved correlations to {os.path.join(output_dir, corr_filename)}")

# Create a dedicated larger plot for SNR feature correlations
print("\nCreating dedicated feature correlation plot for SNR...")
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
top_n = 20  # Show top 20 features
top_features_snr_large = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
top_feat_names_snr_large = [f[0] for f in top_features_snr_large]
top_feat_corrs_snr_large = [f[1] for f in top_features_snr_large]

y_pos = np.arange(len(top_feat_names_snr_large))
colors = ['red' if c < 0 else 'green' for c in top_feat_corrs_snr_large]
bars = ax.barh(y_pos, top_feat_corrs_snr_large, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_feat_names_snr_large, fontsize=12, ha='right')
ax.set_xlabel(f'Correlation with {performance_metric_label}', fontsize=14, fontweight='bold')
ax.set_title(f'Top {top_n} Features Correlated with {performance_metric_label}', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Add value labels on bars
for i, (bar, corr_val) in enumerate(zip(bars, top_feat_corrs_snr_large)):
    width = bar.get_width()
    label_x = width + (0.01 if width >= 0 else -0.01)
    ax.text(label_x, bar.get_y() + bar.get_height()/2, 
           f'{corr_val:.3f}', 
           ha='left' if width >= 0 else 'right', 
           va='center', fontsize=11, fontweight='bold')

# Add legend
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='Positive correlation (better performance)'),
    Patch(facecolor='red', alpha=0.7, label='Negative correlation (worse performance)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"feature_correlations_{performance_metric_name}_detailed.png"), dpi=300, bbox_inches='tight')
print(f"Saved detailed {performance_metric_name} correlation plot to {os.path.join(output_dir, f'feature_correlations_{performance_metric_name}_detailed.png')}")
plt.close()

# Separate visualizations for signal and video features
print("\nCreating separate visualizations for signal and video features...")

# Signal features only
signal_cols = [col for col in signal_df.columns if col in feature_df.columns]
if signal_cols:
    X_signal = feature_df[signal_cols].values
    X_signal = np.nan_to_num(X_signal, nan=0.0)
    X_signal_scaled = StandardScaler().fit_transform(X_signal)
    
    pca_signal = PCA(n_components=2)
    X_signal_pca = pca_signal.fit_transform(X_signal_scaled)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(X_signal_pca[:, 0], X_signal_pca[:, 1], c=performance_metric,
                        cmap='RdYlGn', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax.set_title(f'Signal Features - PCA (Explained Variance: {pca_signal.explained_variance_ratio_.sum():.2%})')
    ax.set_xlabel(f'PC1 ({pca_signal.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_signal.explained_variance_ratio_[1]:.2%})')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label=performance_metric_label)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signal_features_pca.png"), dpi=300, bbox_inches='tight')
    print(f"Saved signal features visualization")

# Video features only
video_cols = [col for col in video_df.columns if col in feature_df.columns]
if video_cols:
    X_video = feature_df[video_cols].values
    X_video = np.nan_to_num(X_video, nan=0.0)
    X_video_scaled = StandardScaler().fit_transform(X_video)
    
    pca_video = PCA(n_components=2)
    X_video_pca = pca_video.fit_transform(X_video_scaled)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(X_video_pca[:, 0], X_video_pca[:, 1], c=performance_metric,
                        cmap='RdYlGn', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax.set_title(f'Video Features - PCA (Explained Variance: {pca_video.explained_variance_ratio_.sum():.2%})')
    ax.set_xlabel(f'PC1 ({pca_video.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_video.explained_variance_ratio_[1]:.2%})')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label=performance_metric_label)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "video_features_pca.png"), dpi=300, bbox_inches='tight')
    print(f"Saved video features visualization")

# ============================================================================
# Create second set of visualizations using HR Absolute Error as performance metric
# ============================================================================
print("\n" + "="*80)
print("Creating visualizations colored by HR Absolute Error...")
print("="*80)

# Compute HR absolute error as performance metric (lower is better)
# Invert for visualization so lower error = higher value (better)
hr_error_metric = feature_df['hr_abs_diff'].values  # Negative so lower error = better
hr_error_metric = np.nan_to_num(hr_error_metric, nan=0.0)
hr_error_label = 'HR Absolute Error (inverted, lower is better)'

# Visualization with HR error
print("\nCreating HR error-based visualizations...")

fig, axes = plt.subplots(2, 2 if HAS_UMAP else 1, figsize=(16, 12) if HAS_UMAP else (16, 8))
if not HAS_UMAP:
    axes = [axes]
else:
    axes = axes.flatten()

# PCA plot
scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=hr_error_metric, 
                            cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
axes[0].set_title(f'PCA Visualization - HR Error (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0], label=hr_error_label)

# t-SNE plot
scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=hr_error_metric,
                            cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
axes[1].set_title('t-SNE Visualization - HR Error')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1], label=hr_error_label)

# UMAP plot (if available)
if HAS_UMAP:
    scatter = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=hr_error_metric,
                                cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[2].set_title('UMAP Visualization - HR Error')
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2], label=hr_error_label)

# Feature correlation with HR error
axes_idx = 3 if HAS_UMAP else 1
if HAS_UMAP:
    axes[axes_idx].axis('off')
    axes_idx = 3
else:
    axes_idx = 1

# Top correlated features with HR error
correlations_hr = {}
for col in feature_cols:
    if col in feature_df.columns:
        corr = np.corrcoef(feature_df[col].fillna(0), hr_error_metric)[0, 1]
        if not np.isnan(corr):
            correlations_hr[col] = corr

top_features_hr = sorted(correlations_hr.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
top_feat_names_hr = [f[0] for f in top_features_hr]
top_feat_corrs_hr = [f[1] for f in top_features_hr]

y_pos = np.arange(len(top_feat_names_hr))
colors = ['red' if c < 0 else 'green' for c in top_feat_corrs_hr]
bars = axes[axes_idx].barh(y_pos, top_feat_corrs_hr, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
axes[axes_idx].set_yticks(y_pos)
axes[axes_idx].set_yticklabels(top_feat_names_hr, fontsize=12, ha='right', fontweight='bold')
axes[axes_idx].set_xlabel(f'Correlation with HR Absolute Error', fontsize=13, fontweight='bold')
axes[axes_idx].set_title('Top 15 Features Correlated with HR Error', fontsize=14, fontweight='bold', pad=15)
axes[axes_idx].grid(True, alpha=0.3, axis='x')
axes[axes_idx].axvline(x=0, color='black', linestyle='--', linewidth=1)

# Add feature names and correlation values on bars
for i, (bar, corr_val, feat_name) in enumerate(zip(bars, top_feat_corrs_hr, top_feat_names_hr)):
    width = bar.get_width()
    # Add correlation value
    label_x = width + (0.02 if width >= 0 else -0.02)
    axes[axes_idx].text(label_x, bar.get_y() + bar.get_height()/2, 
                       f'{corr_val:.3f}', 
                       ha='left' if width >= 0 else 'right', 
                       va='center', fontsize=11, fontweight='bold')
    # Add feature name inside bar if there's space
    if abs(width) > 0.1:  # Only if bar is wide enough
        name_x = width * 0.5  # Center of bar
        axes[axes_idx].text(name_x, bar.get_y() + bar.get_height()/2,
                           feat_name,
                           ha='center', va='center', fontsize=9,
                           color='white', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_visualization_hr_error.png"), dpi=300, bbox_inches='tight')
print(f"Saved HR error visualization to {os.path.join(output_dir, 'feature_visualization_hr_error.png')}")

# Save correlation analysis for HR error
corr_hr_col_name = 'correlation_with_hr_abs_diff'
corr_hr_df = pd.DataFrame(list(correlations_hr.items()), columns=['feature', corr_hr_col_name])
corr_hr_df = corr_hr_df.sort_values(corr_hr_col_name, key=abs, ascending=False)
corr_hr_filename = "feature_correlations_with_hr_abs_diff.csv"
corr_hr_df.to_csv(os.path.join(output_dir, corr_hr_filename), index=False)
print(f"Saved HR error correlations to {os.path.join(output_dir, corr_hr_filename)}")

# Create a dedicated larger plot for HR error feature correlations
print("\nCreating dedicated feature correlation plot for HR error...")
print("\n" + "="*80)
print("TOP 15 FEATURES CORRELATED WITH HR ABSOLUTE ERROR:")
print("="*80)
top_n = 20  # Show top 20 features in plot, but print top 15
top_features_hr_large = sorted(correlations_hr.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
top_feat_names_hr_large = [f[0] for f in top_features_hr_large]
top_feat_corrs_hr_large = [f[1] for f in top_features_hr_large]

# Print top 15 feature names clearly
for i, (feat_name, corr_val) in enumerate(top_features_hr_large[:15], 1):
    direction = "↑ Higher error" if corr_val > 0 else "↓ Lower error"
    print(f"{i:2d}. {feat_name:30s} | Correlation: {corr_val:7.4f} | {direction}")
print("="*80 + "\n")

# Save top 15 to text file
with open(os.path.join(output_dir, "top_15_features_hr_error.txt"), 'w') as f:
    f.write("="*80 + "\n")
    f.write("TOP 15 FEATURES CORRELATED WITH HR ABSOLUTE ERROR\n")
    f.write("="*80 + "\n\n")
    f.write("Rank | Feature Name                    | Correlation | Meaning\n")
    f.write("-"*80 + "\n")
    for i, (feat_name, corr_val) in enumerate(top_features_hr_large[:15], 1):
        direction = "Higher error" if corr_val > 0 else "Lower error"
        f.write(f"{i:4d} | {feat_name:30s} | {corr_val:11.4f} | {direction}\n")
    f.write("\n" + "="*80 + "\n")
    f.write("Note: Positive correlation means higher feature value = higher HR error (worse)\n")
    f.write("      Negative correlation means higher feature value = lower HR error (better)\n")
    f.write("="*80 + "\n")
print(f"Saved top 15 features list to {os.path.join(output_dir, 'top_15_features_hr_error.txt')}")

fig, ax = plt.subplots(1, 1, figsize=(14, 12))  # Made even larger
y_pos = np.arange(len(top_feat_names_hr_large))
colors = ['red' if c < 0 else 'green' for c in top_feat_corrs_hr_large]
bars = ax.barh(y_pos, top_feat_corrs_hr_large, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_feat_names_hr_large, fontsize=14, ha='right', fontweight='bold')  # Increased font size
ax.set_xlabel('Correlation with HR Absolute Error', fontsize=16, fontweight='bold')
ax.set_title(f'Top {top_n} Features Correlated with HR Absolute Error\n(Feature names shown on Y-axis)', 
            fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Add value labels on bars with feature names
for i, (bar, corr_val, feat_name) in enumerate(zip(bars, top_feat_corrs_hr_large, top_feat_names_hr_large)):
    width = bar.get_width()
    label_x = width + (0.01 if width >= 0 else -0.01)
    # Add correlation value
    ax.text(label_x, bar.get_y() + bar.get_height()/2, 
           f'{corr_val:.3f}', 
           ha='left' if width >= 0 else 'right', 
           va='center', fontsize=12, fontweight='bold')
    # Add rank number on the left
    ax.text(-0.05 if width >= 0 else 0.05, bar.get_y() + bar.get_height()/2,
           f"#{i+1}",
           ha='right' if width >= 0 else 'left',
           va='center', fontsize=11, fontweight='bold', color='black')

# Add legend
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='Positive correlation (higher error)'),
    Patch(facecolor='red', alpha=0.7, label='Negative correlation (lower error)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_correlations_hr_error_detailed.png"), dpi=300, bbox_inches='tight')
print(f"Saved detailed HR error correlation plot to {os.path.join(output_dir, 'feature_correlations_hr_error_detailed.png')}")
plt.close()

# Separate visualizations for signal and video features with HR error
print("\nCreating separate HR error visualizations for signal and video features...")

# Signal features only - HR error
signal_cols = [col for col in signal_df.columns if col in feature_df.columns]
if signal_cols:
    X_signal = feature_df[signal_cols].values
    X_signal = np.nan_to_num(X_signal, nan=0.0)
    X_signal_scaled = StandardScaler().fit_transform(X_signal)
    
    pca_signal = PCA(n_components=2)
    X_signal_pca = pca_signal.fit_transform(X_signal_scaled)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(X_signal_pca[:, 0], X_signal_pca[:, 1], c=hr_error_metric,
                        cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax.set_title(f'Signal Features - PCA - HR Error (Explained Variance: {pca_signal.explained_variance_ratio_.sum():.2%})')
    ax.set_xlabel(f'PC1 ({pca_signal.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_signal.explained_variance_ratio_[1]:.2%})')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label=hr_error_label)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signal_features_pca_hr_error.png"), dpi=300, bbox_inches='tight')
    print(f"Saved signal features HR error visualization")

# Video features only - HR error
video_cols = [col for col in video_df.columns if col in feature_df.columns]
if video_cols:
    X_video = feature_df[video_cols].values
    X_video = np.nan_to_num(X_video, nan=0.0)
    X_video_scaled = StandardScaler().fit_transform(X_video)
    
    pca_video = PCA(n_components=2)
    X_video_pca = pca_video.fit_transform(X_video_scaled)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(X_video_pca[:, 0], X_video_pca[:, 1], c=hr_error_metric,
                        cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax.set_title(f'Video Features - PCA - HR Error (Explained Variance: {pca_video.explained_variance_ratio_.sum():.2%})')
    ax.set_xlabel(f'PC1 ({pca_video.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_video.explained_variance_ratio_[1]:.2%})')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label=hr_error_label)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "video_features_pca_hr_error.png"), dpi=300, bbox_inches='tight')
    print(f"Saved video features HR error visualization")

print("\n" + "="*80)
print("Analysis complete!")
print(f"Results saved to: {output_dir}")
print("="*80)



# %%
