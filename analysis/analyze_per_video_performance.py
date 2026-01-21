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
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, linregress
from scipy.stats import gaussian_kde

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

def plot_regression_analysis(df, save_path):
    """Plot regression analysis: predicted vs ground truth HR with R² and correlation."""
    print("Generating regression plot with R-squared...")
    plt.figure(figsize=(10, 8))
    
    # Calculate R² and Pearson correlation
    r2 = r2_score(df["gt_hr"], df["pred_hr"])
    pearson_corr, p_value = pearsonr(df["gt_hr"], df["pred_hr"])
    
    # Calculate regression line
    slope, intercept, r_value, p_val, std_err = linregress(df["gt_hr"], df["pred_hr"])
    line_x = np.array([df["gt_hr"].min(), df["gt_hr"].max()])
    line_y = slope * line_x + intercept
    
    # Create density-colored scatter plot
    xy = np.vstack([df["gt_hr"], df["pred_hr"]])
    z = gaussian_kde(xy)(xy)
    sc = plt.scatter(df["gt_hr"], df["pred_hr"], c=z, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Plot regression line
    plt.plot(line_x, line_y, 'r--', linewidth=2, label=f'Regression line: y = {slope:.2f}x + {intercept:.2f}')
    
    # Plot perfect agreement line (y=x)
    plt.plot(line_x, line_x, 'k--', linewidth=1.5, label='Perfect agreement (y=x)')
    
    plt.xlabel("Ground Truth HR (BPM)", fontsize=12)
    plt.ylabel("Predicted HR (BPM)", fontsize=12)
    plt.title(f"Regression Plot: Predicted vs Ground Truth HR\n$R^2$ = {r2:.3f}, Pearson r = {pearson_corr:.3f} (p = {p_value:.2e})", fontsize=14)
    
    # Add statistics text box
    stats_text = f'$R^2$ = {r2:.3f}\nPearson r = {pearson_corr:.3f}\np-value = {p_value:.2e}\nSlope = {slope:.3f}\nIntercept = {intercept:.2f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend(fontsize=10)
    plt.colorbar(sc, label='Point Density')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "regression_r2_plot.pdf"), dpi=300)
    plt.close()
    print(f"Saved plot: {os.path.join(save_path, 'regression_r2_plot.pdf')}")

def save_analysis_reports(video_stats, df, save_path):
    """Save detailed analysis reports as CSV files."""
    # Save full video statistics
    video_stats.to_csv(os.path.join(save_path, "video_statistics.csv"), index=False)
    print(f"Saved: {os.path.join(save_path, 'video_statistics.csv')}")
    
    # Save worst 20 videos
    worst_20 = video_stats.head(20)
    worst_20.to_csv(os.path.join(save_path, "worst_20_videos.csv"), index=False)
    print(f"Saved: {os.path.join(save_path, 'worst_20_videos.csv')}")
    
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
        
        # Regression statistics
        r2 = r2_score(df["gt_hr"], df["pred_hr"])
        pearson_corr, p_value = pearsonr(df["gt_hr"], df["pred_hr"])
        slope, intercept, r_value, p_val, std_err = linregress(df["gt_hr"], df["pred_hr"])
        f.write("REGRESSION STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"R-squared (R²): {r2:.4f}\n")
        f.write(f"Pearson correlation: {pearson_corr:.4f} (p = {p_value:.2e})\n")
        f.write(f"Regression line: y = {slope:.3f}x + {intercept:.2f}\n\n")
        
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
csv_path = Path("/home/yj167/Desktop/rPPG-Toolbox/scripts/test_runs/exp/PURE_ClipLength240_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_SizeW72_SizeH72/saved_test_outputs/UBFC_UBFC_PURE_tscan_per_chunk_metrics.csv")

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
plot_relative_error_distribution(df, analysis_dir)
plot_regression_analysis(df, analysis_dir)

# Save reports
print("\nSaving analysis reports...")
save_analysis_reports(video_stats, df, analysis_dir)

print("\n" + "=" * 70)
print("Analysis complete!")
print(f"All results saved to: {analysis_dir}")
print("=" * 70)



# %%
