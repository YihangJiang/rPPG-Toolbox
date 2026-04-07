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

# Import functions from analysis package (with fallback for running as script from analysis/)
try:
    from analysis.feature_extraction import extract_signal_features, extract_video_features
    from analysis.data_loading import (
        load_test_data, find_video_cache_file, build_file_list_cache,
        build_metrics_lookup, get_chunk_metrics, create_chunk_info, print_metrics_matching_summary
    )
    from analysis.visualize_standardized_frames import normalize_for_display
except ImportError:
    from feature_extraction import extract_signal_features, extract_video_features
    from data_loading import (
        load_test_data, find_video_cache_file, build_file_list_cache,
        build_metrics_lookup, get_chunk_metrics, create_chunk_info, print_metrics_matching_summary
    )
    try:
        from visualize_standardized_frames import normalize_for_display
    except ImportError:
        normalize_for_display = None

# Try to import umap, but make it optional
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available. Install with: pip install umap-learn")


def run_high_dim_feature_analysis(
    test_results_dir,
    output_dir=None,
    base_cached_path=None,
    exp_data_name=None,
    max_chunks=None,
    skip_frame_visualization=False,
):
    """Run the full high-dimensional feature analysis pipeline.

    Loads test results, extracts signal and video features, runs PCA/t-SNE/UMAP,
    and saves visualizations and correlation CSVs. Optionally visualizes sample
    video frames from the cached dataset.

    Args:
        test_results_dir: Directory containing saved_test_outputs (pickle + optional per_chunk CSV).
        output_dir: Where to write feature_analysis outputs. Default: test_results_dir/feature_analysis.
        base_cached_path: Base path to DATASET_PRE for video chunks and file list. If None, video features skipped.
        exp_data_name: Experiment subfolder name in DATASET_PRE. Default: basename(test_results_dir).
        max_chunks: Limit number of chunks to process (None = all).
        skip_frame_visualization: If True, skip the video frame visualization at the end.
    """
    test_results_dir = os.path.abspath(os.path.expanduser(test_results_dir))
    # Root folder for all feature analysis outputs for this experiment
    if output_dir is None:
        feature_root = os.path.join(test_results_dir, "feature_analysis")
    else:
        feature_root = os.path.abspath(output_dir)
    if exp_data_name is None:
        exp_data_name = os.path.basename(test_results_dir)

    if base_cached_path:
        base_cached_path = os.path.abspath(os.path.expanduser(base_cached_path))
        cached_path = os.path.join(base_cached_path, exp_data_name)
        print(f"EXP_DATA_NAME: {exp_data_name}")
        print(f"Inferred cached_path: {cached_path}")
        if not os.path.exists(cached_path):
            print(f"WARNING: Cached path does not exist: {cached_path}")
            if os.path.exists(base_cached_path):
                subdirs = [d for d in os.listdir(base_cached_path)
                           if os.path.isdir(os.path.join(base_cached_path, d))]
                if subdirs:
                    print(f"  Available subdirectories in {base_cached_path}:")
                    for subdir in sorted(subdirs)[:10]:
                        print(f"    - {subdir}")
                    if len(subdirs) > 10:
                        print(f"    ... and {len(subdirs) - 10} more")
        else:
            print(f"✓ Cached path exists: {cached_path}")

    print("Loading test data...")
    predictions, labels, metrics_df, metadata = load_test_data(test_results_dir)
    fs = metadata['fs']
    run_info = metadata.get("run_info", {}) if isinstance(metadata, dict) else {}

    # Build a naming prefix when we know train/test/model, e.g. UBFC_PURE_TSCAN
    run_prefix = None
    if (
        isinstance(run_info, dict)
        and run_info.get("train_dataset")
        and run_info.get("test_dataset")
        and run_info.get("model_name")
    ):
        run_prefix = f"{run_info['train_dataset']}_{run_info['test_dataset']}_{run_info['model_name']}"

    # Final output directory for feature visualizations
    if run_prefix:
        output_dir = os.path.join(feature_root, f"{run_prefix}_feature")
    else:
        output_dir = feature_root
    os.makedirs(output_dir, exist_ok=True)

    if metrics_df is not None:
        print(f"Found {len(metrics_df)} chunks in metrics CSV")
    print(f"Sampling frequency: {fs} Hz")

    if base_cached_path:
        print("\nBuilding file list cache...")
        file_list_cache = build_file_list_cache(base_cached_path, exp_data_name)
        print(f"Found {len(file_list_cache)} cached files in file lists")
    else:
        file_list_cache = {}

    print("\nExtracting features...")
    signal_features_list = []
    video_features_list = []
    chunk_info = []
    metrics_lookup = build_metrics_lookup(metrics_df)

    print(f"\nProcessing all chunks from pickle file...")
    total_chunks = sum(len(chunks) for chunks in labels.values())
    print(f"Total chunks to process: {total_chunks}")

    sample_video_ids = list(labels.keys())[:3]
    print(f"Sample video_ids from pickle: {sample_video_ids}")
    for vid_id in sample_video_ids:
        sample_chunks = list(labels[vid_id].keys())[:3]
        print(f"  Video {vid_id}: chunk indices {sample_chunks} (total: {len(labels[vid_id])} chunks)")

    chunk_count = 0
    metrics_found_count = 0
    metrics_not_found_count = 0
    sample_not_found_keys = []

    total_to_process = min(max_chunks, total_chunks) if max_chunks else total_chunks
    with tqdm(total=total_to_process, desc="Processing chunks") as pbar:
        for video_id in labels.keys():
            for chunk_idx in labels[video_id].keys():
                if max_chunks and chunk_count >= max_chunks:
                    break

                video_id_str = str(video_id)
                chunk_idx_int = int(chunk_idx)

                signal_data = labels[video_id][chunk_idx].numpy() if hasattr(labels[video_id][chunk_idx], 'numpy') else np.array(labels[video_id][chunk_idx])
                sig_feat = extract_signal_features(signal_data, fs=fs)
                signal_features_list.append(sig_feat)

                if base_cached_path and file_list_cache:
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
                else:
                    video_features_list.append({})

                metrics = get_chunk_metrics(video_id_str, chunk_idx_int, metrics_lookup)
                key = (video_id_str, chunk_idx_int)
                if key in metrics_lookup:
                    metrics_found_count += 1
                else:
                    metrics_not_found_count += 1
                    if len(sample_not_found_keys) < 5:
                        sample_not_found_keys.append(key)

                chunk_info.append(create_chunk_info(video_id_str, chunk_idx_int, metrics))
                chunk_count += 1
                pbar.update(1)

            if max_chunks and chunk_count >= max_chunks:
                break

    print_metrics_matching_summary(metrics_found_count, metrics_not_found_count,
                                  chunk_count, sample_not_found_keys, metrics_lookup)

    print("\nConverting to DataFrames...")
    print(f"  Signal features extracted: {len(signal_features_list)} chunks")
    print(f"  Video features extracted: {len(video_features_list)} chunks")
    print(f"  Chunk info: {len(chunk_info)} chunks")

    if len(signal_features_list) == 0:
        raise ValueError("No signal features extracted! Check that labels are loaded correctly from pickle file.")

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

    if len(video_df.columns) > 0:
        feature_df = pd.concat([signal_df, video_df], axis=1)
    else:
        print("  Warning: No video features found. Proceeding with signal features only.")
        feature_df = signal_df.copy()

    feature_df = pd.concat([info_df, feature_df], axis=1)

    # Add hr_error column (same as hr_abs_diff) to the right of chunk_index; for CSV reference only, not used in TSN/PCA/ranking
    feature_df.insert(
        feature_df.columns.get_loc('chunk_index') + 1,
        'hr_error',
        feature_df['hr_abs_diff'].values
    )

    feature_source = {}
    for col in info_df.columns:
        feature_source[col] = 'info'
    for col in signal_df.columns:
        feature_source[col] = 'signal'
    for col in video_df.columns:
        feature_source[col] = 'video'

    print(f"  Combined feature DF shape: {feature_df.shape}")

    non_meta_cols = [col for col in feature_df.columns if col not in ['video_id', 'chunk_index', 'hr_error', 'SNR', 'MACC', 'gt_hr', 'pred_hr', 'hr_diff', 'hr_abs_diff']]
    feature_df = feature_df.dropna(how='all', subset=non_meta_cols)

    print(f"\nExtracted features from {len(feature_df)} chunks")
    print(f"Signal features: {len(signal_df.columns)}")
    print(f"Video features: {len(video_df.columns)}")

    if len(feature_df) == 0:
        raise ValueError("No valid feature data extracted! Check that predictions and labels are loaded correctly.")

    feature_cols = [col for col in feature_df.columns
                    if col not in ['video_id', 'chunk_index', 'hr_error', 'SNR', 'MACC', 'gt_hr', 'pred_hr', 'hr_diff', 'hr_abs_diff']]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found! Check that feature extraction is working correctly.")

    print(f"Using {len(feature_cols)} feature columns for analysis")

    feature_df.to_csv(os.path.join(output_dir, "extracted_features.csv"), index=False)
    print(f"\nSaved features to {os.path.join(output_dir, 'extracted_features.csv')}")

    X = feature_df[feature_cols].values

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"Feature matrix is empty! Shape: {X.shape}. Check feature extraction.")

    print(f"Feature matrix shape: {X.shape}")

    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        nan_mask = np.isnan(col_data)
        if np.any(nan_mask):
            median_val = np.nanmedian(col_data)
            if np.isnan(median_val):
                median_val = 0.0
            X[nan_mask, col_idx] = median_val
            print(f"  Replaced {np.sum(nan_mask)} NaN values in column {feature_cols[col_idx]} with {median_val}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nPerforming dimensionality reduction...")

    print("  Computing PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    print("  Computing t-SNE (this may take a while)...")
    n_samples = len(X_scaled)
    perplexity = min(30, max(5, n_samples - 1))
    print(f"    Using perplexity={perplexity} for {n_samples} samples")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_scaled)

    if HAS_UMAP:
        print("  Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)

    print("\n" + "="*80)
    print("Creating visualizations colored by HR Absolute Error...")
    print("="*80)

    hr_error_metric = feature_df['hr_abs_diff'].values
    hr_error_metric = np.nan_to_num(hr_error_metric, nan=0.0)
    hr_error_label = 'HR Absolute Error (lower is better)'

    print("\nCreating HR error-based visualizations...")

    if HAS_UMAP:
        # 4 panels: PCA, t-SNE, UMAP, top-correlated features
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        bar_ax = axes[3]
    else:
        # 3 panels: PCA, t-SNE, top-correlated features
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        axes = np.array(axes).ravel()
        bar_ax = axes[2]

    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=hr_error_metric,
                             cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[0].set_title(f'PCA Visualization - HR Error (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label=hr_error_label)

    scatter = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=hr_error_metric,
                              cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[1].set_title('t-SNE Visualization - HR Error')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label=hr_error_label)

    if HAS_UMAP:
        scatter = axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=hr_error_metric,
                                 cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
        axes[2].set_title('UMAP Visualization - HR Error')
        axes[2].set_xlabel('UMAP 1')
        axes[2].set_ylabel('UMAP 2')
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[2], label=hr_error_label)

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
    bars = bar_ax.barh(y_pos, top_feat_corrs_hr, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    bar_ax.set_yticks(y_pos)
    bar_ax.set_yticklabels(top_feat_names_hr, fontsize=12, ha='right', fontweight='bold')
    bar_ax.set_xlabel('Correlation with HR Absolute Error', fontsize=13, fontweight='bold')
    bar_ax.set_title('Top 15 Features Correlated with HR Error', fontsize=14, fontweight='bold', pad=15)
    bar_ax.grid(True, alpha=0.3, axis='x')
    bar_ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    for i, (bar, corr_val, feat_name) in enumerate(zip(bars, top_feat_corrs_hr, top_feat_names_hr)):
        width = bar.get_width()
        label_x = width + (0.02 if width >= 0 else -0.02)
        bar_ax.text(label_x, bar.get_y() + bar.get_height()/2,
                    f'{corr_val:.3f}',
                    ha='left' if width >= 0 else 'right',
                    va='center', fontsize=11, fontweight='bold')
        if abs(width) > 0.1:
            name_x = width * 0.5
            bar_ax.text(name_x, bar.get_y() + bar.get_height()/2,
                        feat_name,
                        ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_visualization_hr_error.png"), dpi=300, bbox_inches='tight')
    print(f"Saved HR error visualization to {os.path.join(output_dir, 'feature_visualization_hr_error.png')}")
    plt.close()

    # Helper to build HR-error figure for a subset of features (signal or video only)
    def _plot_hr_error_by_subset(subset_cols, suffix, title_prefix):
        if len(subset_cols) == 0:
            print(f"  Skipping {suffix} HR error plot (no {suffix} features).")
            return
        X_sub = feature_df[subset_cols].values
        X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
        var_mask = np.var(X_sub, axis=0) > 1e-10
        if np.sum(var_mask) == 0:
            print(f"  Skipping {suffix} HR error plot (all constant).")
            return
        X_sub = X_sub[:, var_mask]
        subset_cols_filtered = [c for c, m in zip(subset_cols, var_mask) if m]
        X_sub_scaled = StandardScaler().fit_transform(X_sub)
        pca_sub = PCA(n_components=2)
        X_pca_sub = pca_sub.fit_transform(X_sub_scaled)
        n_s = len(X_sub_scaled)
        perp = min(30, max(5, n_s - 1))
        tsne_sub = TSNE(n_components=2, random_state=42, perplexity=perp)
        X_tsne_sub = tsne_sub.fit_transform(X_sub_scaled)
        if HAS_UMAP:
            reducer_sub = umap.UMAP(n_components=2, random_state=42)
            X_umap_sub = reducer_sub.fit_transform(X_sub_scaled)
        corr_sub = {}
        for col in subset_cols_filtered:
            if col in feature_df.columns:
                c = np.corrcoef(feature_df[col].fillna(0), hr_error_metric)[0, 1]
                if not np.isnan(c):
                    corr_sub[col] = c
        top_sub = sorted(corr_sub.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        top_names = [f[0] for f in top_sub]
        top_vals = [f[1] for f in top_sub]
        if HAS_UMAP:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            idx = 3
        else:
            fig, axes = plt.subplots(1, 3, figsize=(24, 6))
            axes = np.array(axes).ravel()
            idx = 2
        axes[0].scatter(X_pca_sub[:, 0], X_pca_sub[:, 1], c=hr_error_metric, cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
        axes[0].set_title(f'{title_prefix} PCA - HR Error ({pca_sub.explained_variance_ratio_.sum():.2%})')
        axes[0].set_xlabel(f'PC1 ({pca_sub.explained_variance_ratio_[0]:.2%})')
        axes[0].set_ylabel(f'PC2 ({pca_sub.explained_variance_ratio_[1]:.2%})')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(axes[0].collections[0], ax=axes[0], label=hr_error_label)
        axes[1].scatter(X_tsne_sub[:, 0], X_tsne_sub[:, 1], c=hr_error_metric, cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
        axes[1].set_title(f'{title_prefix} t-SNE - HR Error')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(axes[1].collections[0], ax=axes[1], label=hr_error_label)
        if HAS_UMAP:
            axes[2].scatter(X_umap_sub[:, 0], X_umap_sub[:, 1], c=hr_error_metric, cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
            axes[2].set_title(f'{title_prefix} UMAP - HR Error')
            axes[2].set_xlabel('UMAP 1')
            axes[2].set_ylabel('UMAP 2')
            axes[2].grid(True, alpha=0.3)
            plt.colorbar(axes[2].collections[0], ax=axes[2], label=hr_error_label)
        y_pos = np.arange(len(top_names))
        colors_bar = ['red' if c < 0 else 'green' for c in top_vals]
        axes[idx].barh(y_pos, top_vals, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(top_names, fontsize=12, ha='right', fontweight='bold')
        axes[idx].set_xlabel('Correlation with HR Absolute Error', fontsize=13, fontweight='bold')
        axes[idx].set_title(f'{title_prefix} - Top 15 Correlated with HR Error', fontsize=14, fontweight='bold', pad=15)
        axes[idx].grid(True, alpha=0.3, axis='x')
        axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=1)
        for bar, cv, fn in zip(axes[idx].patches, top_vals, top_names):
            w = bar.get_width()
            axes[idx].text(w + (0.02 if w >= 0 else -0.02), bar.get_y() + bar.get_height()/2, f'{cv:.3f}', ha='left' if w >= 0 else 'right', va='center', fontsize=11, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"feature_visualization_hr_error_{suffix}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved {suffix} HR error visualization to {out_path}")
        plt.close()

    signal_cols = [c for c in signal_df.columns if c in feature_df.columns]
    video_cols = [c for c in video_df.columns if c in feature_df.columns]
    print("\nCreating signal-only and video-only HR error visualizations...")
    _plot_hr_error_by_subset(signal_cols, "signal", "Signal Features")
    _plot_hr_error_by_subset(video_cols, "video", "Video Features")

    corr_hr_col_name = 'correlation_with_hr_abs_diff'
    corr_hr_df = pd.DataFrame(list(correlations_hr.items()), columns=['feature', corr_hr_col_name])
    corr_hr_df['source'] = corr_hr_df['feature'].map(feature_source).fillna('unknown')
    corr_hr_df = corr_hr_df.sort_values(corr_hr_col_name, key=abs, ascending=False)
    corr_hr_df = corr_hr_df[['feature', 'source', corr_hr_col_name]]
    corr_hr_filename = "feature_correlations_with_hr_abs_diff.csv"
    corr_hr_df.to_csv(os.path.join(output_dir, corr_hr_filename), index=False)
    print(f"Saved HR error correlations to {os.path.join(output_dir, corr_hr_filename)}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


def visualize_dataset_pre_frames(base_cached_path, exp_data_name, file_list_cache,
                                 output_dir, num_videos=5, frames_per_video=5):
    """Visualize sample frames from one chunk per video in DATASET_PRE.

    Args:
        base_cached_path: Base path to DATASET_PRE
        exp_data_name: Experiment data name (subdirectory in DATASET_PRE)
        file_list_cache: Cache of file paths from build_file_list_cache
        output_dir: Directory to save visualizations
        num_videos: Number of videos to visualize
        frames_per_video: Number of frames to show per chunk
    """
    if normalize_for_display is None:
        raise ImportError("normalize_for_display from visualize_standardized_frames is required.")

    # Get unique video IDs from file list cache
    video_ids = set()
    for (vid_id, chunk_idx), file_path in file_list_cache.items():
        video_ids.add(vid_id)

    video_ids = sorted(list(video_ids))[:num_videos]
    print(f"\nVisualizing frames from {len(video_ids)} videos (one chunk each): {video_ids}")

    all_visualized = False
    video_chunk_shape = None

    for video_idx, video_id in enumerate(video_ids):
        # One chunk per video: take first chunk only
        video_chunks = [(vid, ch) for (vid, ch) in file_list_cache.keys() if vid == video_id]
        video_chunks = sorted(video_chunks, key=lambda x: int(x[1]))[:1]

        if not video_chunks:
            print(f"  No chunks found for video {video_id}")
            continue

        (vid_id, chunk_idx_int) = video_chunks[0]
        video_file = find_video_cache_file(base_cached_path, str(video_id),
                                          int(chunk_idx_int), file_list_cache, exp_data_name)

        if not video_file or not os.path.exists(video_file):
            print(f"  Warning: Could not find video file for {video_id}_chunk{chunk_idx_int}")
            continue

        try:
            video_chunk = np.load(video_file)  # Shape: (T, H, W, C)
            video_chunk_shape = video_chunk.shape
            print(f"\n  Processing video {video_id}, chunk {chunk_idx_int}: shape {video_chunk.shape}")

            num_frames = video_chunk.shape[0]
            frame_indices = np.linspace(0, num_frames - 1, frames_per_video, dtype=int)
            num_channels = video_chunk.shape[-1] if video_chunk.ndim > 2 else 1

            # If num_channels > 3, one row per group of 3 channels; otherwise one row
            num_channel_groups = max(1, (num_channels + 2) // 3)
            n_rows = num_channel_groups
            n_cols = frames_per_video

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            if n_cols == 1:
                axes = axes.reshape(-1, 1)

            for frame_col, frame_idx in enumerate(frame_indices):
                frame = video_chunk[frame_idx]  # (H, W, C)

                for row in range(n_rows):
                    ax = axes[row, frame_col]
                    c_start = row * 3
                    c_end = min(c_start + 3, num_channels)

                    if num_channels == 1:
                        slab = frame[:, :, 0] if frame.ndim > 2 else frame
                        display_frame = normalize_for_display(slab.astype(np.float64))
                        ax.imshow(display_frame, cmap='gray')
                        ax.set_title(f'Frame {frame_idx}\n1ch', fontsize=8)
                    elif num_channels <= 3:
                        slab = frame[:, :, :3]
                        display_frame = normalize_for_display(slab.astype(np.float64))
                        ax.imshow(display_frame)
                        ax.set_title(f'Frame {frame_idx}', fontsize=9)
                    else:
                        slab = frame[:, :, c_start:c_end]
                        if slab.shape[-1] == 3:
                            display_frame = normalize_for_display(slab.astype(np.float64))
                            ax.imshow(display_frame)
                        else:
                            # Pad to 3 for consistent display (e.g. 1 or 2 channels left)
                            pad = np.zeros((*slab.shape[:2], 3), dtype=slab.dtype)
                            pad[:, :, :slab.shape[-1]] = slab
                            display_frame = normalize_for_display(pad.astype(np.float64))
                            ax.imshow(display_frame)
                        ch_info = f'ch {c_start}-{c_end - 1}' if num_channel_groups > 1 else ''
                        ax.set_title(f'Frame {frame_idx}\n{ch_info}'.strip(), fontsize=8)
                    ax.axis('off')

            shape_str = str(video_chunk_shape) if video_chunk_shape else "Unknown"
            fig.suptitle(f'Video {video_id} (chunk {chunk_idx_int})\nShape: {shape_str}',
                         fontsize=12, y=0.995)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'video_{video_id}_frames.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
            plt.close()
            all_visualized = True

        except Exception as e:
            print(f"  Error processing video {video_id}: {e}")
            continue

    if all_visualized:
        print(f"\n✓ Video frame visualizations saved to: {output_dir}")
    else:
        print(f"\n⚠ No video frames were visualized. Check file paths and cache.")

if __name__ == "__main__":
    test_results_dir = os.path.expanduser(
        "/home/yj167/Desktop/rPPG-Toolbox/scripts/test_runs/exp/PURE_ClipLength240_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_SizeW96_SizeH96"
    )
    base_cached_path = "/mnt/nvme2/rppg_data/DATASET_PRE"
    run_high_dim_feature_analysis(
        test_results_dir=test_results_dir,
        base_cached_path=base_cached_path,
        max_chunks=None,
        skip_frame_visualization=False,
    )

# %%
