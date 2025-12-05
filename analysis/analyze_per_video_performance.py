"""
Post-processing analysis script for rPPG model evaluation results.
This script analyzes per-chunk metrics to identify videos with worst performance.
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_and_validate_data(csv_path):
    """Load CSV and validate required columns."""
    df = pd.read_csv(csv_path)
    
    # Sanity check
    required_cols = {"video_id", "chunk_index", "gt_hr", "pred_hr", "SNR", "MACC"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns. Expected: {required_cols}, Found: {df.columns.tolist()}")
    
    return df

def compute_errors(df):
    """Compute various error metrics."""
    df["abs_error"] = np.abs(df["gt_hr"] - df["pred_hr"])
    df["relative_error"] = (df["abs_error"] / df["gt_hr"]) * 100  # Percentage error
    df["squared_error"] = (df["gt_hr"] - df["pred_hr"]) ** 2
    return df

def compute_video_statistics(df):
    """Compute statistics per video."""
    video_stats = df.groupby("video_id").agg({
        "abs_error": ["mean", "median", "std", "min", "max"],
        "relative_error": ["mean", "median"],
        "squared_error": "mean",  # This is MSE per video
        "gt_hr": "mean",
        "pred_hr": "mean",
        "SNR": "mean",
        "MACC": "mean",
        "chunk_index": "count"  # Number of chunks per video
    }).reset_index()
    
    # Flatten column names
    video_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in video_stats.columns.values]
    
    # Compute RMSE per video
    video_stats["RMSE"] = np.sqrt(video_stats["squared_error_mean"])
    
    # Sort by median absolute error (descending)
    video_stats = video_stats.sort_values("abs_error_median", ascending=False)
    
    return video_stats

def plot_mean_vs_median_error(video_stats, save_path):
    """Plot mean vs median absolute error per video."""
    x = np.arange(len(video_stats))
    width = 0.35
    
    plt.figure(figsize=(20, 6))
    plt.bar(x - width/2, video_stats["abs_error_mean"], width, 
            label="Mean Error", color="skyblue", edgecolor='black')
    plt.bar(x + width/2, video_stats["abs_error_median"], width, 
            label="Median Error", color="mediumseagreen", edgecolor='black')
    
    # Format video IDs for clarity
    video_labels = video_stats["video_id"].astype(str)
    
    # Rotate and space xticks for readability
    plt.xticks(x, video_labels, rotation=90, ha='right', fontsize=8)
    
    plt.xlabel("Video ID (sorted by median error)", fontsize=12)
    plt.ylabel("Absolute HR Error (BPM)", fontsize=12)
    plt.title("Per-Video Mean vs. Median Absolute HR Error", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "mean_vs_median_error.pdf"), dpi=300)
    plt.close()
    print(f"Saved plot: {os.path.join(save_path, 'mean_vs_median_error.pdf')}")

def plot_rmse_per_video(video_stats, save_path):
    """Plot RMSE per video."""
    plt.figure(figsize=(20, 6))
    x = np.arange(len(video_stats))
    
    plt.bar(x, video_stats["RMSE"], color="coral", edgecolor='black')
    
    video_labels = video_stats["video_id"].astype(str)
    plt.xticks(x, video_labels, rotation=90, ha='right', fontsize=8)
    
    plt.xlabel("Video ID (sorted by median error)", fontsize=12)
    plt.ylabel("RMSE (BPM)", fontsize=12)
    plt.title("Per-Video Root Mean Squared Error", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "rmse_per_video.pdf"), dpi=300)
    plt.close()
    print(f"Saved plot: {os.path.join(save_path, 'rmse_per_video.pdf')}")

def plot_snr_vs_error(video_stats, save_path):
    """Plot SNR vs absolute error to see correlation."""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(video_stats["SNR_mean"], video_stats["abs_error_median"], 
                alpha=0.6, s=80, c=video_stats["abs_error_median"], 
                cmap='YlOrRd', edgecolors='black')
    
    plt.xlabel("Mean SNR", fontsize=12)
    plt.ylabel("Median Absolute Error (BPM)", fontsize=12)
    plt.title("Relationship between SNR and Error", fontsize=14)
    plt.colorbar(label='Median Abs Error (BPM)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "snr_vs_error.pdf"), dpi=300)
    plt.close()
    print(f"Saved plot: {os.path.join(save_path, 'snr_vs_error.pdf')}")

def plot_macc_vs_error(video_stats, save_path):
    """Plot MACC vs absolute error."""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(video_stats["MACC_mean"], video_stats["abs_error_median"], 
                alpha=0.6, s=80, c=video_stats["abs_error_median"], 
                cmap='YlOrRd', edgecolors='black')
    
    plt.xlabel("Mean MACC", fontsize=12)
    plt.ylabel("Median Absolute Error (BPM)", fontsize=12)
    plt.title("Relationship between MACC and Error", fontsize=14)
    plt.colorbar(label='Median Abs Error (BPM)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "macc_vs_error.pdf"), dpi=300)
    plt.close()
    print(f"Saved plot: {os.path.join(save_path, 'macc_vs_error.pdf')}")

def plot_relative_error_distribution(df, save_path):
    """Plot distribution of relative errors."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(df["relative_error"], bins=50, color="steelblue", 
             edgecolor='black', alpha=0.7)
    
    plt.xlabel("Relative Error (%)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Relative Heart Rate Errors", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, "relative_error_distribution.pdf"), dpi=300)
    plt.close()
    print(f"Saved plot: {os.path.join(save_path, 'relative_error_distribution.pdf')}")

def plot_top_worst_videos(video_stats, df, save_path, top_n=10):
    """Plot detailed comparison for top N worst videos."""
    worst_videos = video_stats.head(top_n)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Error metrics for worst videos
    x = np.arange(len(worst_videos))
    width = 0.25
    
    axes[0].bar(x - width, worst_videos["abs_error_mean"], width, 
                label="Mean Abs Error", color="crimson", edgecolor='black')
    axes[0].bar(x, worst_videos["abs_error_median"], width, 
                label="Median Abs Error", color="orange", edgecolor='black')
    axes[0].bar(x + width, worst_videos["RMSE"], width, 
                label="RMSE", color="darkred", edgecolor='black')
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(worst_videos["video_id"], rotation=45, ha='right')
    axes[0].set_ylabel("Error (BPM)", fontsize=12)
    axes[0].set_title(f"Top {top_n} Worst Performing Videos - Error Metrics", fontsize=14)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Plot 2: SNR and MACC for worst videos
    ax2_twin = axes[1].twinx()
    
    axes[1].bar(x - width/2, worst_videos["SNR_mean"], width, 
                label="SNR", color="steelblue", edgecolor='black')
    ax2_twin.bar(x + width/2, worst_videos["MACC_mean"], width, 
                 label="MACC", color="seagreen", edgecolor='black')
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(worst_videos["video_id"], rotation=45, ha='right')
    axes[1].set_ylabel("SNR", fontsize=12, color="steelblue")
    ax2_twin.set_ylabel("MACC", fontsize=12, color="seagreen")
    axes[1].set_title(f"Top {top_n} Worst Performing Videos - Quality Metrics", fontsize=14)
    axes[1].tick_params(axis='y', labelcolor="steelblue")
    ax2_twin.tick_params(axis='y', labelcolor="seagreen")
    axes[1].legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"top_{top_n}_worst_videos.pdf"), dpi=300)
    plt.close()
    print(f"Saved plot: {os.path.join(save_path, f'top_{top_n}_worst_videos.pdf')}")

def save_analysis_reports(video_stats, df, save_path):
    """Save detailed analysis reports as CSV files."""
    # Save full video statistics
    video_stats.to_csv(os.path.join(save_path, "video_statistics.csv"), index=False)
    print(f"Saved: {os.path.join(save_path, 'video_statistics.csv')}")
    
    # Save worst 20 videos
    worst_20 = video_stats.head(20)
    worst_20.to_csv(os.path.join(save_path, "worst_20_videos.csv"), index=False)
    print(f"Saved: {os.path.join(save_path, 'worst_20_videos.csv')}")
    
    # Save best 20 videos
    best_20 = video_stats.tail(20)
    best_20.to_csv(os.path.join(save_path, "best_20_videos.csv"), index=False)
    print(f"Saved: {os.path.join(save_path, 'best_20_videos.csv')}")
    
    # Save detailed chunk analysis for worst videos
    worst_video_ids = video_stats.head(20)["video_id"].tolist()
    worst_chunks = df[df["video_id"].isin(worst_video_ids)].copy()
    worst_chunks = worst_chunks.sort_values(["video_id", "chunk_index"])
    worst_chunks.to_csv(os.path.join(save_path, "worst_videos_chunk_details.csv"), index=False)
    print(f"Saved: {os.path.join(save_path, 'worst_videos_chunk_details.csv')}")
    
    # Generate summary statistics
    with open(os.path.join(save_path, "summary_statistics.txt"), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RPPG MODEL EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total number of videos analyzed: {len(video_stats)}\n")
        f.write(f"Total number of chunks: {len(df)}\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean Absolute Error (all videos): {df['abs_error'].mean():.2f} ± {df['abs_error'].std():.2f} BPM\n")
        f.write(f"Median Absolute Error (all videos): {df['abs_error'].median():.2f} BPM\n")
        f.write(f"RMSE (all chunks): {np.sqrt((df['squared_error']).mean()):.2f} BPM\n")
        f.write(f"Mean Relative Error: {df['relative_error'].mean():.2f}%\n")
        f.write(f"Mean SNR: {df['SNR'].mean():.2f}\n")
        f.write(f"Mean MACC: {df['MACC'].mean():.4f}\n\n")
        
        f.write("TOP 10 WORST PERFORMING VIDEOS:\n")
        f.write("-" * 70 + "\n")
        for idx, row in video_stats.head(10).iterrows():
            f.write(f"{row['video_id']:<15} | MAE: {row['abs_error_mean']:>6.2f} BPM | "
                   f"Median: {row['abs_error_median']:>6.2f} BPM | RMSE: {row['RMSE']:>6.2f} BPM | "
                   f"SNR: {row['SNR_mean']:>7.2f} | MACC: {row['MACC_mean']:>6.4f}\n")
        
        f.write("\nTOP 10 BEST PERFORMING VIDEOS:\n")
        f.write("-" * 70 + "\n")
        for idx, row in video_stats.tail(10).iloc[::-1].iterrows():
            f.write(f"{row['video_id']:<15} | MAE: {row['abs_error_mean']:>6.2f} BPM | "
                   f"Median: {row['abs_error_median']:>6.2f} BPM | RMSE: {row['RMSE']:>6.2f} BPM | "
                   f"SNR: {row['SNR_mean']:>7.2f} | MACC: {row['MACC_mean']:>6.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"Saved: {os.path.join(save_path, 'summary_statistics.txt')}")

# %%
"""Main analysis pipeline."""
# Define paths - using specific CSV file path
csv_path = Path("/hpc/group/dunnlab/rppg_data/rPPG-Toolbox/scripts/test_runs/exp/PURE_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_SizeW72_SizeH72/saved_test_outputs/UBFC_UBFC_PURE_tscan_per_chunk_metrics.csv")

if not csv_path.exists():
    raise FileNotFoundError(f"Could not find metrics file: {csv_path}")

print(f"Analyzing metrics from: {csv_path}")

# Create analysis output directory
analysis_dir = csv_path.parent / "performance_analysis"
analysis_dir.mkdir(exist_ok=True)
print(f"Analysis results will be saved to: {analysis_dir}\n")

# Load and process data
print("Loading data...")
df = load_and_validate_data(csv_path)

print("Computing error metrics...")
df = compute_errors(df)

print("Computing per-video statistics...")
video_stats = compute_video_statistics(df)

# Generate all plots
print("\nGenerating plots...")
plot_mean_vs_median_error(video_stats, analysis_dir)
plot_rmse_per_video(video_stats, analysis_dir)
plot_snr_vs_error(video_stats, analysis_dir)
plot_macc_vs_error(video_stats, analysis_dir)
plot_relative_error_distribution(df, analysis_dir)
plot_top_worst_videos(video_stats, df, analysis_dir, top_n=10)

# Save reports
print("\nSaving analysis reports...")
save_analysis_reports(video_stats, df, analysis_dir)

print("\n" + "=" * 70)
print("Analysis complete!")
print(f"All results saved to: {analysis_dir}")
print("=" * 70)



# %%
