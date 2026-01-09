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

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import functions from separate modules
from feature_extraction import extract_signal_features, extract_video_features
from data_loading import (
    load_test_data, find_video_cache_file, build_file_list_cache,
    build_metrics_lookup, get_chunk_metrics, create_chunk_info, print_metrics_matching_summary
)

# Try to import umap, but make it optional
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available. Install with: pip install umap-learn")

# %%
# Configuration
# Expand ~ to home directory and convert to absolute path
test_results_dir = os.path.expanduser("~/Desktop/rPPG-Toolbox/scripts/test_runs/exp/PURE_ClipLength240_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_SizeW72_SizeH72")
test_results_dir = os.path.abspath(test_results_dir)

# Infer cached path from test_results_dir
# The EXP_DATA_NAME is the last component of test_results_dir
# Cached path should be: BASE_CACHED_PATH / EXP_DATA_NAME
exp_data_name = os.path.basename(test_results_dir)
base_cached_path = "/mnt/nvme2/rppg_data/DATASET_PRE"  # Base path for preprocessed data
cached_path = os.path.join(base_cached_path, exp_data_name)

print(f"EXP_DATA_NAME: {exp_data_name}")
print(f"Inferred cached_path: {cached_path}")
if not os.path.exists(cached_path):
    print(f"WARNING: Cached path does not exist: {cached_path}")
    print(f"  The code will search in subdirectories of: {base_cached_path}")
    # List available subdirectories to help user find the correct path
    if os.path.exists(base_cached_path):
        subdirs = [d for d in os.listdir(base_cached_path) 
                   if os.path.isdir(os.path.join(base_cached_path, d))]
        if subdirs:
            print(f"  Available subdirectories in {base_cached_path}:")
            for subdir in sorted(subdirs)[:10]:  # Show first 10
                print(f"    - {subdir}")
            if len(subdirs) > 10:
                print(f"    ... and {len(subdirs) - 10} more")
else:
    print(f"✓ Cached path exists: {cached_path}")

# %%
output_dir = os.path.join(test_results_dir, "feature_analysis")
os.makedirs(output_dir, exist_ok=True)

print("Loading test data...")
predictions, labels, metrics_df, metadata = load_test_data(test_results_dir)
fs = metadata['fs']

if metrics_df is not None:
    print(f"Found {len(metrics_df)} chunks in metrics CSV")
print(f"Sampling frequency: {fs} Hz")

# Build file list cache for faster lookup
# DataFileLists are stored in DATASET_PRE/DataFileLists (not in EXP_DATA_NAME subdirectory)
# Filter CSV files to only process the one matching this experiment's EXP_DATA_NAME
print("\nBuilding file list cache...")
file_list_cache = build_file_list_cache(base_cached_path, exp_data_name)
print(f"Found {len(file_list_cache)} cached files in file lists")

# %%
# Extract features for each chunk
print("\nExtracting features...")
signal_features_list = []
video_features_list = []
chunk_info = []

# Build a lookup for metrics by (video_id, chunk_index)
metrics_lookup = build_metrics_lookup(metrics_df)

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
            # DataFileLists are in DATASET_PRE/DataFileLists, actual files may be in DATASET_PRE/EXP_DATA_NAME
            video_file = find_video_cache_file(base_cached_path, video_id_str, chunk_idx_int, file_list_cache, exp_data_name)
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
            metrics = get_chunk_metrics(video_id_str, chunk_idx_int, metrics_lookup)
            
            # Track metrics matching
            key = (video_id_str, chunk_idx_int)
            if key in metrics_lookup:
                metrics_found_count += 1
            else:
                metrics_not_found_count += 1
                if len(sample_not_found_keys) < 5:
                    sample_not_found_keys.append(key)
            
            # Store chunk info with metrics
            chunk_info.append(create_chunk_info(video_id_str, chunk_idx_int, metrics))
            
            chunk_count += 1
            pbar.update(1)
        
        if max_chunks and chunk_count >= max_chunks:
            break

# Print metrics matching summary
print_metrics_matching_summary(metrics_found_count, metrics_not_found_count, 
                               chunk_count, sample_not_found_keys, metrics_lookup)

# %%
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

# Track feature sources for correlation analysis
feature_source = {}
# Info features (metrics and metadata)
for col in info_df.columns:
    feature_source[col] = 'info'
# Signal features
for col in signal_df.columns:
    feature_source[col] = 'signal'
# Video features
for col in video_df.columns:
    feature_source[col] = 'video'

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

# ============================================================================
# Create visualizations using HR Absolute Error as performance metric
# ============================================================================
print("\n" + "="*80)
print("Creating visualizations colored by HR Absolute Error...")
print("="*80)

# Compute HR absolute error as performance metric (lower is better)
hr_error_metric = feature_df['hr_abs_diff'].values
hr_error_metric = np.nan_to_num(hr_error_metric, nan=0.0)
hr_error_label = 'HR Absolute Error (lower is better)'

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
# Add source column to indicate where each feature comes from
corr_hr_df['source'] = corr_hr_df['feature'].map(feature_source).fillna('unknown')
corr_hr_df = corr_hr_df.sort_values(corr_hr_col_name, key=abs, ascending=False)
# Reorder columns: feature, source, correlation
corr_hr_df = corr_hr_df[['feature', 'source', corr_hr_col_name]]
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
# ============================================================================
# Visualize video frames from DATASET_PRE
# ============================================================================
print("\n" + "="*80)
print("Visualizing video frames from DATASET_PRE...")
print("="*80)

def visualize_dataset_pre_frames(base_cached_path, exp_data_name, file_list_cache, 
                                  output_dir, num_videos=5, frames_per_video=5, 
                                  chunks_per_video=2):
    """Visualize sample frames from preprocessed video chunks in DATASET_PRE.
    
    Args:
        base_cached_path: Base path to DATASET_PRE
        exp_data_name: Experiment data name (subdirectory in DATASET_PRE)
        file_list_cache: Cache of file paths from build_file_list_cache
        output_dir: Directory to save visualizations
        num_videos: Number of videos to visualize
        frames_per_video: Number of frames to show per chunk
        chunks_per_video: Number of chunks to show per video
    """
    # Get unique video IDs from file list cache
    video_ids = set()
    for (vid_id, chunk_idx), file_path in file_list_cache.items():
        video_ids.add(vid_id)
    
    video_ids = sorted(list(video_ids))[:num_videos]
    print(f"\nVisualizing frames from {len(video_ids)} videos: {video_ids}")
    
    all_visualized = False
    video_chunk_shape = None  # Track shape for title
    
    for video_idx, video_id in enumerate(video_ids):
        # Get chunks for this video
        video_chunks = [(vid, ch) for (vid, ch) in file_list_cache.keys() 
                       if vid == video_id]
        video_chunks = sorted(video_chunks, key=lambda x: int(x[1]))[:chunks_per_video]
        
        if not video_chunks:
            print(f"  No chunks found for video {video_id}")
            continue
        
        print(f"\n  Processing video {video_id}: {len(video_chunks)} chunks")
        
        # Create figure for this video
        fig, axes = plt.subplots(chunks_per_video, frames_per_video, 
                                 figsize=(frames_per_video * 2.5, chunks_per_video * 2.5))
        if chunks_per_video == 1:
            axes = axes.reshape(1, -1)
        if frames_per_video == 1:
            axes = axes.reshape(-1, 1)
        
        for chunk_idx, (vid_id, chunk_idx_int) in enumerate(video_chunks):
            # Load video chunk
            video_file = find_video_cache_file(base_cached_path, str(video_id), 
                                              int(chunk_idx_int), file_list_cache, exp_data_name)
            
            if not video_file or not os.path.exists(video_file):
                print(f"    Warning: Could not find video file for {video_id}_chunk{chunk_idx_int}")
                continue
            
            try:
                video_chunk = np.load(video_file)  # Shape: (T, H, W, C)
                video_chunk_shape = video_chunk.shape  # Store for title
                print(f"    Loaded chunk {chunk_idx_int}: shape {video_chunk.shape}")
                
                # Sample frames evenly across the chunk
                num_frames = video_chunk.shape[0]
                frame_indices = np.linspace(0, num_frames - 1, frames_per_video, dtype=int)
                
                for frame_col, frame_idx in enumerate(frame_indices):
                    frame = video_chunk[frame_idx]  # Shape: (H, W, C)
                    ax = axes[chunk_idx, frame_col]
                    
                    # Handle different channel configurations
                    num_channels = frame.shape[-1] if len(frame.shape) > 2 else 1
                    
                    if num_channels == 3:
                        # RGB visualization
                        display_frame = frame.copy()
                        # Normalize for display if needed (handle standardized/diff-normalized)
                        if frame.min() < 0 or frame.max() > 255:
                            display_frame = (display_frame - display_frame.min()) / (display_frame.max() - display_frame.min() + 1e-8)
                            display_frame = np.clip(display_frame, 0, 1)
                        else:
                            display_frame = display_frame / 255.0
                        
                        ax.imshow(display_frame)
                        ax.set_title(f'Chunk {chunk_idx_int}\nFrame {frame_idx}', fontsize=9)
                    elif num_channels > 3:
                        # Multiple transformations concatenated
                        # Show first 3 channels (usually Raw RGB) if available
                        if num_channels >= 3:
                            display_frame = frame[:, :, :3].copy()
                            if frame.min() < 0 or frame.max() > 255:
                                display_frame = (display_frame - display_frame.min()) / (display_frame.max() - display_frame.min() + 1e-8)
                                display_frame = np.clip(display_frame, 0, 1)
                            else:
                                display_frame = display_frame / 255.0
                            
                            ax.imshow(display_frame)
                            ch_info = f'{num_channels}ch (showing RGB)'
                        else:
                            # Grayscale single channel
                            display_frame = frame[:, :, 0].copy()
                            if frame.min() < 0 or frame.max() > 255:
                                display_frame = (display_frame - display_frame.min()) / (display_frame.max() - display_frame.min() + 1e-8)
                                display_frame = np.clip(display_frame, 0, 1)
                            else:
                                display_frame = display_frame / 255.0
                            
                            ax.imshow(display_frame, cmap='gray')
                            ch_info = f'{num_channels}ch (grayscale)'
                        
                        ax.set_title(f'Chunk {chunk_idx_int}\nFrame {frame_idx}\n{ch_info}', fontsize=8)
                    else:
                        # Single channel - grayscale
                        display_frame = frame[:, :, 0] if len(frame.shape) > 2 else frame
                        display_frame = display_frame.copy()
                        if frame.min() < 0 or frame.max() > 255:
                            display_frame = (display_frame - display_frame.min()) / (display_frame.max() - display_frame.min() + 1e-8)
                            display_frame = np.clip(display_frame, 0, 1)
                        else:
                            display_frame = display_frame / 255.0
                        
                        ax.imshow(display_frame, cmap='gray')
                        ax.set_title(f'Chunk {chunk_idx_int}\nFrame {frame_idx}\n1ch', fontsize=8)
                    
                    ax.axis('off')
                
            except Exception as e:
                print(f"    Error loading chunk {chunk_idx_int}: {e}")
                continue
        
        # Add overall title
        shape_str = str(video_chunk_shape) if video_chunk_shape else "Unknown"
        fig.suptitle(f'Video {video_id}\nShape: {shape_str}', 
                     fontsize=12, y=0.995)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(output_dir, f'video_{video_id}_frames.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {save_path}")
        plt.close()
        
        all_visualized = True
    
    if all_visualized:
        print(f"\n✓ Video frame visualizations saved to: {output_dir}")
    else:
        print(f"\n⚠ No video frames were visualized. Check file paths and cache.")

# Call the visualization function
if len(file_list_cache) > 0:
    visualize_dataset_pre_frames(
        base_cached_path=base_cached_path,
        exp_data_name=exp_data_name,
        file_list_cache=file_list_cache,
        output_dir=output_dir,
        num_videos=5,  # Visualize 5 videos
        frames_per_video=5,  # 5 frames per chunk
        chunks_per_video=2  # 2 chunks per video
    )
else:
    print("\n⚠ No video files found in cache. Cannot visualize frames.")
    print(f"   File list cache size: {len(file_list_cache)}")

print("\n" + "="*80)
print("Video frame visualization complete!")
print("="*80)

# %%
