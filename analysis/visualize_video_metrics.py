"""
High-dimensional feature visualization for rPPG model performance analysis.
Simplified version that works with per-video metrics instead of per-chunk.

This script extracts features from:
1. Raw signal labels (PPG/BVP signals) - frequency domain, statistical, temporal features
2. Raw video data - optical flow, color statistics, motion, texture features

Features are visualized using dimensionality reduction (PCA, t-SNE, UMAP)
and correlated with model performance metrics.
"""

import pickle
import pandas as pd
import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
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
            features['dominant_freq'] = pos_freqs[hr_mask][np.argmax(pos_fft[hr_mask])]
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
        very_low = (pos_freqs >= 0.04) & (pos_freqs < 0.15)
        low = (pos_freqs >= 0.15) & (pos_freqs < 0.4)
        high = (pos_freqs >= 0.4) & (pos_freqs < 0.6)
        hr_band = (pos_freqs >= 0.5) & (pos_freqs <= 4.0)
        
        features['power_vlf'] = np.sum(pos_fft[very_low]) if np.any(very_low) else 0
        features['power_lf'] = np.sum(pos_fft[low]) if np.any(low) else 0
        features['power_hf'] = np.sum(pos_fft[high]) if np.any(high) else 0
        features['power_hr'] = np.sum(pos_fft[hr_band]) if np.any(hr_band) else 0
        features['lf_hf_ratio'] = features['power_lf'] / (features['power_hf'] + 1e-10)
    else:
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

def load_test_data(test_results_dir):
    """Load test results from pickle and per-video CSV files.
    
    Args:
        test_results_dir: Directory containing test results
        
    Returns:
        tuple: (predictions_dict, labels_dict, metrics_df, metadata)
    """
    saved_outputs_dir = os.path.join(test_results_dir, 'saved_test_outputs')
    
    if not os.path.exists(saved_outputs_dir):
        raise FileNotFoundError(f"Directory not found: {saved_outputs_dir}")
    
    # Find pickle file
    pickle_files = [f for f in os.listdir(saved_outputs_dir) if f.endswith('.pickle')]
    if not pickle_files:
        raise FileNotFoundError(f"No pickle file found in {saved_outputs_dir}")
    pickle_path = os.path.join(saved_outputs_dir, pickle_files[0])
    
    # Load pickle
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    predictions = data['predictions']
    labels = data['labels']
    fs = data.get('fs', 30)
    label_type = data.get('label_type', 'DiffNormalized')
    
    # Load per-VIDEO metrics CSV (not per-chunk)
    metrics_files = [f for f in os.listdir(saved_outputs_dir) 
                     if 'per_video' in f and f.endswith('.csv')]
    if not metrics_files:
        raise FileNotFoundError(f"No per_video metrics CSV found in {saved_outputs_dir}")
    metrics_path = os.path.join(saved_outputs_dir, metrics_files[0])
    metrics_df = pd.read_csv(metrics_path)
    
    return predictions, labels, metrics_df, {'fs': fs, 'label_type': label_type}

# Configuration
test_results_dir = os.path.expanduser("~/Desktop/rPPG-Toolbox/scripts/test_runs/exp/PURE_ClipLength240_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_SizeW72_SizeH72")
test_results_dir = os.path.abspath(test_results_dir)

output_dir = os.path.join(test_results_dir, "feature_analysis_per_video")
os.makedirs(output_dir, exist_ok=True)

print("Loading test data...")
predictions, labels, metrics_df, metadata = load_test_data(test_results_dir)
fs = metadata['fs']

print(f"Found {len(metrics_df)} videos in metrics")
print(f"Sampling frequency: {fs} Hz")
print(f"Columns in metrics: {list(metrics_df.columns)}")

# Extract features for each VIDEO (concatenate all chunks per video)
print("\nExtracting features per video...")
signal_features_list = []
video_info = []

for idx, row in tqdm(metrics_df.iterrows(), total=len(metrics_df), desc="Processing videos"):
    video_id = str(row['video_id'])
    
    # Concatenate all chunks for this video
    if video_id in labels:
        # Get all chunks and concatenate
        chunk_indices = sorted(labels[video_id].keys())
        all_chunks = []
        for chunk_idx in chunk_indices:
            chunk_data = labels[video_id][chunk_idx]
            chunk_array = chunk_data.numpy() if hasattr(chunk_data, 'numpy') else np.array(chunk_data)
            all_chunks.append(chunk_array.flatten())
        
        # Concatenate all chunks into one long signal
        full_signal = np.concatenate(all_chunks)
        
        # Extract signal features from full video signal
        sig_feat = extract_signal_features(full_signal, fs=fs)
        signal_features_list.append(sig_feat)
        
        # Store video info with metrics
        video_info.append({
            'video_id': video_id,
            'num_chunks': len(chunk_indices),
            'signal_length': len(full_signal),
            'avg_SNR': row.get('avg_SNR', np.nan),
            'avg_MACC': row.get('avg_MACC', np.nan),
            'avg_gt_hr_fft': row.get('avg_gt_hr_fft', np.nan),
            'avg_pred_hr_fft': row.get('avg_pred_hr_fft', np.nan),
        })
    else:
        print(f"Warning: Video {video_id} not found in labels")
        signal_features_list.append({})
        video_info.append({
            'video_id': video_id,
            'num_chunks': 0,
            'signal_length': 0,
            'avg_SNR': row.get('avg_SNR', np.nan),
            'avg_MACC': row.get('avg_MACC', np.nan),
            'avg_gt_hr_fft': row.get('avg_gt_hr_fft', np.nan),
            'avg_pred_hr_fft': row.get('avg_pred_hr_fft', np.nan),
        })

# Convert to DataFrames
print("\nConverting to DataFrames...")
signal_df = pd.DataFrame(signal_features_list)
info_df = pd.DataFrame(video_info)

print(f"  Signal DF shape: {signal_df.shape}")
print(f"  Info DF shape: {info_df.shape}")

# Combine features
feature_df = pd.concat([info_df, signal_df], axis=1)

print(f"  Combined feature DF shape: {feature_df.shape}")

# Remove rows with all NaN features
feature_cols_check = [col for col in signal_df.columns]
feature_df = feature_df.dropna(how='all', subset=feature_cols_check)

print(f"\nExtracted features from {len(feature_df)} videos")
print(f"Signal features: {len(signal_df.columns)}")

# Save feature dataframe
feature_df.to_csv(os.path.join(output_dir, "extracted_features_per_video.csv"), index=False)
print(f"\nSaved features to {os.path.join(output_dir, 'extracted_features_per_video.csv')}")

# Prepare features for dimensionality reduction
feature_cols = [col for col in feature_df.columns 
                if col not in ['video_id', 'num_chunks', 'signal_length', 
                              'avg_SNR', 'avg_MACC', 'avg_gt_hr_fft', 'avg_pred_hr_fft']]

print(f"Using {len(feature_cols)} feature columns for analysis")

# Extract feature matrix
X = feature_df[feature_cols].values

print(f"Feature matrix shape: {X.shape}")

# Handle NaN and inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Remove constant features
feature_variance = np.var(X, axis=0)
non_constant_mask = feature_variance > 1e-10
if np.sum(~non_constant_mask) > 0:
    removed_features = [feature_cols[i] for i in range(len(feature_cols)) if not non_constant_mask[i]]
    print(f"  Removing {len(removed_features)} constant features: {removed_features}")
    X = X[:, non_constant_mask]
    feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if non_constant_mask[i]]

print(f"Final feature matrix shape: {X.shape}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Performance metric for coloring
performance_metric = feature_df['avg_SNR'].values
performance_metric = np.nan_to_num(performance_metric, nan=0.0)

# Dimensionality reduction
print("\nPerforming dimensionality reduction...")

# PCA
print("  Computing PCA...")
pca = PCA(n_components=min(2, X_scaled.shape[1]))
X_pca = pca.fit_transform(X_scaled)
print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# t-SNE
print("  Computing t-SNE...")
n_samples = len(X_scaled)
perplexity = min(30, max(5, n_samples - 1))
print(f"    Using perplexity={perplexity} for {n_samples} samples")
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP (if available)
if HAS_UMAP:
    print("  Computing UMAP...")
    n_neighbors = min(15, n_samples - 1)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
    X_umap = reducer.fit_transform(X_scaled)

# Visualization
print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# PCA plot
scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=performance_metric, 
                            cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black', linewidths=1)
axes[0].set_title(f'PCA Visualization - Per Video (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0], label='Avg SNR')

# Annotate some points
for i, row in feature_df.iterrows():
    if i % 10 == 0:  # Annotate every 10th video
        axes[0].annotate(row['video_id'], (X_pca[i, 0], X_pca[i, 1]), 
                        fontsize=8, alpha=0.7)

# t-SNE plot
scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=performance_metric,
                            cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black', linewidths=1)
axes[1].set_title('t-SNE Visualization - Per Video')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1], label='Avg SNR')

# UMAP plot (if available)
if HAS_UMAP:
    scatter = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=performance_metric,
                                cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black', linewidths=1)
    axes[2].set_title('UMAP Visualization - Per Video')
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2], label='Avg SNR')
else:
    axes[2].axis('off')
    axes[2].text(0.5, 0.5, 'UMAP not available\nInstall with: pip install umap-learn',
                ha='center', va='center', transform=axes[2].transAxes)

# Feature correlation with performance
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
axes[3].barh(y_pos, top_feat_corrs, color=colors, alpha=0.7)
axes[3].set_yticks(y_pos)
axes[3].set_yticklabels(top_feat_names, fontsize=9)
axes[3].set_xlabel('Correlation with Avg SNR')
axes[3].set_title('Top 15 Features Correlated with Performance')
axes[3].grid(True, alpha=0.3, axis='x')
axes[3].axvline(x=0, color='black', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_visualization_per_video.png"), dpi=300, bbox_inches='tight')
print(f"Saved visualization to {os.path.join(output_dir, 'feature_visualization_per_video.png')}")

# Save correlation analysis
corr_df = pd.DataFrame(list(correlations.items()), columns=['feature', 'correlation_with_avg_SNR'])
corr_df = corr_df.sort_values('correlation_with_avg_SNR', key=abs, ascending=False)
corr_df.to_csv(os.path.join(output_dir, "feature_correlations_per_video.csv"), index=False)
print(f"Saved correlations to {os.path.join(output_dir, 'feature_correlations_per_video.csv')}")

print("\n" + "="*80)
print("Analysis complete!")
print(f"Results saved to: {output_dir}")
print("="*80)

