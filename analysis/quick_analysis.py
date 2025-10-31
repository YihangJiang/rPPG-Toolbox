"""
Quick analysis script - can be run with command line argument
Usage: python quick_analysis.py <path_to_csv>
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def analyze_csv(csv_path, output_dir=None):
    """Analyze per-chunk metrics CSV file."""
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path), "performance_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Sanity check
    if not {"video_id", "gt_hr", "pred_hr"}.issubset(df.columns):
        raise ValueError("Missing required columns: video_id, gt_hr, pred_hr")
    
    # Compute absolute error
    df["abs_error"] = np.abs(df["gt_hr"] - df["pred_hr"])
    
    # Group by video and compute both mean and median
    video_stats = df.groupby("video_id")["abs_error"].agg(["mean", "median"]).reset_index()
    video_stats = video_stats.sort_values("median", ascending=False)
    
    # Save video statistics
    video_stats.to_csv(os.path.join(output_dir, "video_error_stats.csv"), index=False)
    print(f"Saved: {os.path.join(output_dir, 'video_error_stats.csv')}")
    
    # Plotting setup
    x = np.arange(len(video_stats))
    width = 0.35
    
    plt.figure(figsize=(20, 6))
    plt.bar(x - width/2, video_stats["mean"], width, label="Mean Error", color="skyblue", edgecolor='black')
    plt.bar(x + width/2, video_stats["median"], width, label="Median Error", color="mediumseagreen", edgecolor='black')
    
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
    
    plot_path = os.path.join(output_dir, "mean_vs_median_error.pdf")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")
    
    # Print worst 10 videos
    print("\n" + "="*60)
    print("TOP 10 WORST PERFORMING VIDEOS:")
    print("="*60)
    for idx, row in video_stats.head(10).iterrows():
        print(f"{row['video_id']:<15} | Mean: {row['mean']:>6.2f} BPM | Median: {row['median']:>6.2f} BPM")
    
    print("\n" + "="*60)
    print("TOP 10 BEST PERFORMING VIDEOS:")
    print("="*60)
    for idx, row in video_stats.tail(10).iloc[::-1].iterrows():
        print(f"{row['video_id']:<15} | Mean: {row['mean']:>6.2f} BPM | Median: {row['median']:>6.2f} BPM")
    
    print("\n" + "="*60)
    print(f"Overall Mean Absolute Error: {df['abs_error'].mean():.2f} BPM")
    print(f"Overall Median Absolute Error: {df['abs_error'].median():.2f} BPM")
    print("="*60)
    
    print(f"\nAll results saved to: {output_dir}")

#%%
if len(sys.argv) < 2:
    print("Usage: python quick_analysis.py <path_to_csv>")
    sys.exit(1)

csv_path = sys.argv[1]
if not os.path.exists(csv_path):
    print(f"Error: File not found: {csv_path}")
    sys.exit(1)

analyze_csv(csv_path)


# %%
