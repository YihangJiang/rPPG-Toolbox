"""
Simple analysis script for current rPPG test results.
Just run: python analyze_current_results.py
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set paths
csv_path = "/hpc/group/dunnlab/rppg_data/rPPG-Toolbox/test_runs/exp/UBFC-PHYS_ClipLength700_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceFalse_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse_SizeW72_SizeH72/saved_test_outputs/_per_chunk_metrics.csv"
output_dir = os.path.join(os.path.dirname(csv_path), "..", "performance_analysis")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
df = pd.read_csv(csv_path)

# Sanity check
if not {"video_id", "gt_hr", "pred_hr"}.issubset(df.columns):
    raise ValueError("Missing required columns: video_id, gt_hr, pred_hr")

# Compute absolute error
df["abs_error"] = np.abs(df["gt_hr"] - df["pred_hr"])
df["relative_error"] = (df["abs_error"] / df["gt_hr"]) * 100

# Group by video and compute statistics
print("Computing per-video statistics...")
video_stats = df.groupby("video_id").agg({
    "abs_error": ["mean", "median", "std"],
    "relative_error": ["mean", "median"],
    "gt_hr": "mean",
    "pred_hr": "mean",
    "SNR": "mean",
    "MACC": "mean",
    "chunk_index": "count"
}).reset_index()

# Flatten column names
video_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                       for col in video_stats.columns.values]

# Sort by median absolute error (descending)
video_stats = video_stats.sort_values("abs_error_median", ascending=False)

# Save video statistics
video_stats.to_csv(os.path.join(output_dir, "video_statistics.csv"), index=False)
print(f"Saved: {os.path.join(output_dir, 'video_statistics.csv')}")

# Save worst 20 videos
worst_20 = video_stats.head(20)
worst_20.to_csv(os.path.join(output_dir, "worst_20_videos.csv"), index=False)
print(f"Saved: {os.path.join(output_dir, 'worst_20_videos.csv')}")

# Save best 20 videos
best_20 = video_stats.tail(20)
best_20.to_csv(os.path.join(output_dir, "best_20_videos.csv"), index=False)
print(f"Saved: {os.path.join(output_dir, 'best_20_videos.csv')}")

# Plot 1: Mean vs Median Error
print("\nGenerating plots...")
x = np.arange(len(video_stats))
width = 0.35

plt.figure(figsize=(20, 6))
plt.bar(x - width/2, video_stats["abs_error_mean"], width, 
        label="Mean Error", color="skyblue", edgecolor='black')
plt.bar(x + width/2, video_stats["abs_error_median"], width, 
        label="Median Error", color="mediumseagreen", edgecolor='black')

video_labels = video_stats["video_id"].astype(str)
plt.xticks(x, video_labels, rotation=90, ha='right', fontsize=8)

plt.xlabel("Video ID (sorted by median error)", fontsize=12)
plt.ylabel("Absolute HR Error (BPM)", fontsize=12)
plt.title("Per-Video Mean vs. Median Absolute HR Error", fontsize=14)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mean_vs_median_error.pdf"), dpi=300)
plt.close()
print(f"Saved: {os.path.join(output_dir, 'mean_vs_median_error.pdf')}")

# Plot 2: SNR vs Error
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
plt.savefig(os.path.join(output_dir, "snr_vs_error.pdf"), dpi=300)
plt.close()
print(f"Saved: {os.path.join(output_dir, 'snr_vs_error.pdf')}")

# Plot 3: MACC vs Error
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
plt.savefig(os.path.join(output_dir, "macc_vs_error.pdf"), dpi=300)
plt.close()
print(f"Saved: {os.path.join(output_dir, 'macc_vs_error.pdf')}")

# Plot 4: Top 10 worst videos detailed comparison
worst_10 = video_stats.head(10)
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

x = np.arange(len(worst_10))
width = 0.25

# Error metrics
axes[0].bar(x - width, worst_10["abs_error_mean"], width, 
            label="Mean Abs Error", color="crimson", edgecolor='black')
axes[0].bar(x, worst_10["abs_error_median"], width, 
            label="Median Abs Error", color="orange", edgecolor='black')
axes[0].bar(x + width, worst_10["abs_error_std"], width, 
            label="Std Dev", color="darkred", edgecolor='black')

axes[0].set_xticks(x)
axes[0].set_xticklabels(worst_10["video_id"], rotation=45, ha='right')
axes[0].set_ylabel("Error (BPM)", fontsize=12)
axes[0].set_title("Top 10 Worst Performing Videos - Error Metrics", fontsize=14)
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.6)

# Quality metrics
ax2_twin = axes[1].twinx()

axes[1].bar(x - width/2, worst_10["SNR_mean"], width, 
            label="SNR", color="steelblue", edgecolor='black')
ax2_twin.bar(x + width/2, worst_10["MACC_mean"], width, 
             label="MACC", color="seagreen", edgecolor='black')

axes[1].set_xticks(x)
axes[1].set_xticklabels(worst_10["video_id"], rotation=45, ha='right')
axes[1].set_ylabel("SNR", fontsize=12, color="steelblue")
ax2_twin.set_ylabel("MACC", fontsize=12, color="seagreen")
axes[1].set_title("Top 10 Worst Performing Videos - Quality Metrics", fontsize=14)
axes[1].tick_params(axis='y', labelcolor="steelblue")
ax2_twin.tick_params(axis='y', labelcolor="seagreen")
axes[1].legend(loc='upper left')
ax2_twin.legend(loc='upper right')
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_10_worst_videos.pdf"), dpi=300)
plt.close()
print(f"Saved: {os.path.join(output_dir, 'top_10_worst_videos.pdf')}")

# Print summary
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)
print(f"Total videos analyzed: {len(video_stats)}")
print(f"Total chunks: {len(df)}")
print(f"\nOverall Mean Absolute Error: {df['abs_error'].mean():.2f} ± {df['abs_error'].std():.2f} BPM")
print(f"Overall Median Absolute Error: {df['abs_error'].median():.2f} BPM")
print(f"Mean Relative Error: {df['relative_error'].mean():.2f}%")
print(f"Mean SNR: {df['SNR'].mean():.2f}")
print(f"Mean MACC: {df['MACC'].mean():.4f}")

print("\n" + "="*70)
print("TOP 10 WORST PERFORMING VIDEOS:")
print("="*70)
for idx, row in video_stats.head(10).iterrows():
    print(f"{row['video_id']:<15} | Mean: {row['abs_error_mean']:>6.2f} BPM | "
          f"Median: {row['abs_error_median']:>6.2f} BPM | SNR: {row['SNR_mean']:>7.2f} | "
          f"MACC: {row['MACC_mean']:>6.4f}")

print("\n" + "="*70)
print("TOP 10 BEST PERFORMING VIDEOS:")
print("="*70)
for idx, row in video_stats.tail(10).iloc[::-1].iterrows():
    print(f"{row['video_id']:<15} | Mean: {row['abs_error_mean']:>6.2f} BPM | "
          f"Median: {row['abs_error_median']:>6.2f} BPM | SNR: {row['SNR_mean']:>7.2f} | "
          f"MACC: {row['MACC_mean']:>6.4f}")

print("\n" + "="*70)
print(f"All results saved to: {output_dir}")
print("="*70)


# %%
