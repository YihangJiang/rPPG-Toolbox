# %%
"""
Plot distribution of pixel values for a video chunk from DATASET_PRE.

Loads a preprocessed video chunk (.npy) and visualizes:
- Histogram of all pixel values
- Channel-wise distributions (if multiple channels)
- Statistics (mean, std, min, max, percentiles)
- Frame-wise statistics over time
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy import stats

# %%
# Configuration: specify path to a video chunk in DATASET_PRE
# Example: "/mnt/nvme2/rppg_data/DATASET_PRE/EXP_NAME/subject1_input0.npy"
VIDEO_CHUNK_PATH = None  # Set this to your video chunk path

# Or specify base path and experiment name to pick a random chunk
DATASET_PRE_BASE = "/mnt/nvme2/rppg_data/DATASET_PRE"
EXP_DATA_NAME = ""  # Leave empty to randomly select an experiment folder
PICK_RANDOM = True  # If True, picks a random chunk from the experiment

# %%
def find_random_experiment(dataset_pre_base):
    """Find a random experiment folder in DATASET_PRE that contains video chunks."""
    import glob
    if not os.path.exists(dataset_pre_base):
        raise FileNotFoundError(f"DATASET_PRE base path not found: {dataset_pre_base}")
    
    # Get all subdirectories (experiment folders)
    exp_folders = [d for d in os.listdir(dataset_pre_base) 
                   if os.path.isdir(os.path.join(dataset_pre_base, d))]
    
    if not exp_folders:
        raise FileNotFoundError(f"No experiment folders found in {dataset_pre_base}")
    
    # Filter to folders that contain *_input*.npy files
    valid_exps = []
    for exp_name in exp_folders:
        exp_path = os.path.join(dataset_pre_base, exp_name)
        pattern = os.path.join(exp_path, "*_input*.npy")
        chunks = glob.glob(pattern)
        if chunks:
            valid_exps.append(exp_name)
    
    if not valid_exps:
        raise FileNotFoundError(f"No experiment folders with video chunks found in {dataset_pre_base}")
    
    selected = np.random.choice(valid_exps)
    print(f"Found {len(valid_exps)} experiment folders. Selected: {selected}")
    return selected


def find_random_chunk(dataset_pre_base, exp_data_name):
    """Find a random video chunk in DATASET_PRE."""
    import glob
    exp_path = os.path.join(dataset_pre_base, exp_data_name)
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"Experiment path not found: {exp_path}")
    pattern = os.path.join(exp_path, "*_input*.npy")
    chunks = glob.glob(pattern)
    if not chunks:
        raise FileNotFoundError(f"No input chunks found in {exp_path}")
    return np.random.choice(chunks)


def load_video_chunk(chunk_path):
    """Load video chunk from .npy file."""
    if not os.path.isfile(chunk_path):
        raise FileNotFoundError(f"Video chunk not found: {chunk_path}")
    data = np.load(chunk_path)
    print(f"Loaded chunk: {chunk_path}")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    return data


def plot_distribution(data, title_suffix=""):
    """
    Plot distribution of pixel values in video chunk.
    
    Args:
        data: Video chunk array (T, H, W, C) or (T, H, W)
        title_suffix: Additional text for plot title
    """
    T, H, W = data.shape[:3]
    C = data.shape[3] if data.ndim == 4 else 1
    
    # Flatten all pixels
    all_pixels = data.flatten()
    
    # Statistics
    stats = {
        'mean': np.mean(all_pixels),
        'std': np.std(all_pixels),
        'min': np.min(all_pixels),
        'max': np.max(all_pixels),
        'median': np.median(all_pixels),
        'p25': np.percentile(all_pixels, 25),
        'p75': np.percentile(all_pixels, 75),
    }
    
    print("\n=== Pixel Value Statistics ===")
    for key, val in stats.items():
        print(f"  {key:8s}: {val:12.6f}")
    
    # Test for normality (Gaussian distribution)
    # Sample a subset if too large (Shapiro-Wilk has limit ~5000)
    sample_size = min(5000, len(all_pixels))
    sample_pixels = np.random.choice(all_pixels, size=sample_size, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(sample_pixels)
    
    # Also compute skewness and kurtosis
    skewness = stats.skew(all_pixels)
    kurtosis = stats.kurtosis(all_pixels)  # excess kurtosis (0 for normal)
    
    print("\n=== Normality Tests ===")
    print(f"  Shapiro-Wilk test (sample n={sample_size}):")
    print(f"    Statistic: {shapiro_stat:.6f}")
    print(f"    p-value: {shapiro_p:.6f}")
    print(f"    {'Likely Gaussian' if shapiro_p > 0.05 else 'NOT Gaussian'} (p > 0.05)")
    print(f"  Skewness: {skewness:.6f} (0 = symmetric, >0 = right tail)")
    print(f"  Excess Kurtosis: {kurtosis:.6f} (0 = normal, >0 = heavy tails)")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Overall histogram
    ax1 = plt.subplot(2, 4, 1)
    ax1.hist(all_pixels, bins=100, alpha=0.7, edgecolor='black')
    ax1.axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.3f}")
    ax1.axvline(stats['median'], color='g', linestyle='--', label=f"Median: {stats['median']:.3f}")
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Overall Distribution{title_suffix}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Channel-wise distributions (if C > 1)
    if C > 1:
        ax2 = plt.subplot(2, 4, 2)
        channel_names = []
        if C == 3:
            channel_names = ['R (or DiffNorm R)', 'G (or DiffNorm G)', 'B (or DiffNorm B)']
        elif C == 6:
            channel_names = ['DiffNorm R', 'DiffNorm G', 'DiffNorm B', 'Std R', 'Std G', 'Std B']
        else:
            channel_names = [f'Channel {i}' for i in range(C)]
        
        for c in range(C):
            channel_pixels = data[:, :, :, c].flatten()
            ax2.hist(channel_pixels, bins=50, alpha=0.6, label=channel_names[c] if c < len(channel_names) else f'Ch{c}')
        ax2.set_xlabel('Pixel Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Channel-wise Distributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2 = plt.subplot(2, 4, 2)
        ax2.text(0.5, 0.5, 'Single channel', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Channel-wise Distributions')
    
    # 3. Frame-wise mean over time
    ax3 = plt.subplot(2, 4, 3)
    frame_means = np.mean(data, axis=(1, 2, 3) if data.ndim == 4 else (1, 2))
    ax3.plot(frame_means, linewidth=1.5)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Mean Pixel Value')
    ax3.set_title('Mean Pixel Value per Frame')
    ax3.grid(True, alpha=0.3)
    
    # 4. Frame-wise std over time
    ax4 = plt.subplot(2, 4, 4)
    frame_stds = np.std(data, axis=(1, 2, 3) if data.ndim == 4 else (1, 2))
    ax4.plot(frame_stds, linewidth=1.5, color='orange')
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Std Pixel Value')
    ax4.set_title('Std Pixel Value per Frame')
    ax4.grid(True, alpha=0.3)
    
    # 5. Box plot by channel (if C > 1)
    if C > 1:
        ax5 = plt.subplot(2, 4, 5)
        channel_data = [data[:, :, :, c].flatten() for c in range(C)]
        bp = ax5.boxplot(channel_data, labels=[channel_names[c] if c < len(channel_names) else f'Ch{c}' for c in range(C)], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax5.set_ylabel('Pixel Value')
        ax5.set_title('Channel-wise Box Plots')
        ax5.grid(True, alpha=0.3, axis='y')
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    else:
        ax5 = plt.subplot(2, 4, 5)
        ax5.boxplot([all_pixels], labels=['All'])
        ax5.set_ylabel('Pixel Value')
        ax5.set_title('Box Plot')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Q-Q plot (to check Gaussian distribution)
    ax6 = plt.subplot(2, 4, 6)
    # Sample for Q-Q plot if too large
    qq_sample_size = min(1000, len(all_pixels))
    qq_sample = np.random.choice(all_pixels, size=qq_sample_size, replace=False)
    stats.probplot(qq_sample, dist="norm", plot=ax6)
    ax6.set_title(f'Q-Q Plot vs Normal\n(Skew={skewness:.3f}, Kurt={kurtosis:.3f})')
    ax6.grid(True, alpha=0.3)
    
    # 7. Histogram with Gaussian overlay (if approximately normal)
    ax7 = plt.subplot(2, 4, 7)
    n, bins, patches = ax7.hist(all_pixels, bins=100, alpha=0.7, density=True, edgecolor='black', label='Data')
    # Overlay theoretical Gaussian
    mu, sigma = stats['mean'], stats['std']
    x_gauss = np.linspace(all_pixels.min(), all_pixels.max(), 200)
    y_gauss = stats.norm.pdf(x_gauss, mu, sigma)
    ax7.plot(x_gauss, y_gauss, 'r-', linewidth=2, label=f'Gaussian(μ={mu:.2f}, σ={sigma:.2f})')
    ax7.set_xlabel('Pixel Value')
    ax7.set_ylabel('Density')
    ax7.set_title('Histogram vs Gaussian Overlay')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Cumulative distribution
    ax8 = plt.subplot(2, 4, 8)
    sorted_pixels = np.sort(all_pixels)
    cumulative = np.arange(1, len(sorted_pixels) + 1) / len(sorted_pixels)
    ax8.plot(sorted_pixels, cumulative, linewidth=2, label='Data CDF')
    # Overlay theoretical Gaussian CDF
    cdf_gauss = stats.norm.cdf(sorted_pixels, mu, sigma)
    ax8.plot(sorted_pixels, cdf_gauss, 'r--', linewidth=2, label='Gaussian CDF')
    ax8.axvline(stats['median'], color='g', linestyle=':', label=f"Median: {stats['median']:.3f}")
    ax8.set_xlabel('Pixel Value')
    ax8.set_ylabel('Cumulative Probability')
    ax8.set_title('CDF vs Gaussian CDF')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, stats


# %%
# Main execution
if __name__ == "__main__" or VIDEO_CHUNK_PATH is not None or PICK_RANDOM:
    # Determine which chunk to load
    if VIDEO_CHUNK_PATH is not None:
        chunk_path = VIDEO_CHUNK_PATH
    elif PICK_RANDOM:
        # If EXP_DATA_NAME is empty, randomly select an experiment folder
        exp_name = EXP_DATA_NAME if EXP_DATA_NAME else find_random_experiment(DATASET_PRE_BASE)
        chunk_path = find_random_chunk(DATASET_PRE_BASE, exp_name)
        print(f"Selected random chunk: {chunk_path}")
    else:
        raise ValueError("Set VIDEO_CHUNK_PATH or set PICK_RANDOM=True")
    
    # Load data
    data = load_video_chunk(chunk_path)
    
    # Plot distribution
    chunk_name = os.path.basename(chunk_path)
    fig, stats = plot_distribution(data, title_suffix=f"\n({chunk_name})")
    
    plt.show()
    
    print(f"\n=== Summary ===")
    print(f"Video chunk: {chunk_name}")
    print(f"Shape: {data.shape}")
    print(f"Total pixels: {data.size:,}")
    print(f"Value range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
    print(f"\nNote: DiffNormalized data divides frame differences by (F[t+1]+F[t])")
    print(f"      then standardizes by global std. This does NOT guarantee Gaussian distribution.")
    print(f"      Check Q-Q plot and normality tests above to verify if it's approximately normal.")

# %%
