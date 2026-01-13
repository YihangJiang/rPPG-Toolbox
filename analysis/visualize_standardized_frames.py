# %%
"""
Visualize standardized video frames from PURE dataset cached in DATASET_PRE.

This script loads preprocessed video frames, extracts DiffNormalized and Standardized
channels, and creates side-by-side comparison visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
from pathlib import Path
from tqdm import tqdm

# Add analysis directory to Python path for imports
try:
    # If running as script, use __file__
    analysis_dir = Path(__file__).parent.absolute()
except NameError:
    # If running in Jupyter notebook, use absolute path
    analysis_dir = Path('/home/yj167/Desktop/rPPG-Toolbox/analysis')

if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))

# Import helper functions from data_loading module
from data_loading import build_file_list_cache, find_video_cache_file


def normalize_for_display(img):
    """Normalize image array for display (handles negative values from standardization).
    
    Args:
        img: Image array (H, W, C) or (H, W)
        
    Returns:
        Normalized image array in range [0, 1]
    """
    img = img.copy()
    img_min = img.min()
    img_max = img.max()
    
    if img_max - img_min < 1e-8:
        # Constant image
        return np.zeros_like(img)
    
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return np.clip(img, 0, 1)


def extract_channels(frame, num_channels):
    """Extract DiffNormalized and Standardized channels from frame.
    
    Args:
        frame: Frame array (H, W, C) where C >= 6
        num_channels: Total number of channels in frame
        
    Returns:
        tuple: (diff_frame, std_frame) where each is (H, W, 3) RGB
    """
    if num_channels == 6:
        # Standard case: channels 0-2 are DiffNormalized, 3-5 are Standardized
        diff_frame = frame[:, :, 0:3]
        std_frame = frame[:, :, 3:6]
    elif num_channels == 3:
        # Only one transformation - assume it's the first one (DiffNormalized)
        diff_frame = frame[:, :, 0:3]
        std_frame = frame[:, :, 0:3]  # Duplicate if only one transformation
        print("Warning: Only 3 channels found, duplicating for comparison")
    else:
        # Multiple transformations - try to extract first and second set of RGB channels
        if num_channels >= 6:
            diff_frame = frame[:, :, 0:3]
            std_frame = frame[:, :, 3:6]
        else:
            # Fallback: use first 3 channels for both
            diff_frame = frame[:, :, 0:3]
            std_frame = frame[:, :, 0:3]
            print(f"Warning: Unexpected channel count {num_channels}, using first 3 channels for both")
    
    return diff_frame, std_frame


def visualize_frame_comparison(video_chunk, video_id, chunk_idx, frame_idx, output_path):
    """Create side-by-side visualization of DiffNormalized vs Standardized frame.
    
    Args:
        video_chunk: Video chunk array (T, H, W, C)
        video_id: Video/subject ID
        chunk_idx: Chunk index
        frame_idx: Frame index within chunk
        output_path: Path to save the image
    """
    # Extract frame
    frame = video_chunk[frame_idx]  # (H, W, C)
    num_channels = frame.shape[-1] if len(frame.shape) > 2 else 1
    
    # Extract channels
    diff_frame, std_frame = extract_channels(frame, num_channels)
    
    # Normalize for display
    diff_display = normalize_for_display(diff_frame)
    std_display = normalize_for_display(std_frame)
    
    # Create side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display DiffNormalized frame
    ax1.imshow(diff_display)
    ax1.set_title('DiffNormalized', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display Standardized frame
    ax2.imshow(std_display)
    ax2.set_title('Standardized', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add overall title
    fig.suptitle(f'Video {video_id} | Chunk {chunk_idx} | Frame {frame_idx}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# %%
"""Main function to visualize standardized frames."""

# Configuration
# Base path to DATASET_PRE
base_cached_path = "/mnt/nvme2/rppg_data/DATASET_PRE"

# EXP_DATA_NAME for PURE dataset with DiffNormalized + Standardized
# This should match the actual cached directory name
# You can modify this or make it auto-detect
exp_data_name = None  # Will auto-detect if None

# Number of samples to visualize
num_samples = 10

print("=" * 60)
print("Visualizing Standardized Frames from PURE Dataset")
print("=" * 60)
print(f"Base cached path: {base_cached_path}")

# Auto-detect EXP_DATA_NAME if not provided
if exp_data_name is None:
    print("\nAuto-detecting EXP_DATA_NAME...")
    if os.path.exists(base_cached_path):
        # Look for PURE dataset directories with DiffNormalized and Standardized
        subdirs = [d for d in os.listdir(base_cached_path) 
                    if os.path.isdir(os.path.join(base_cached_path, d))]
        
        # Filter for PURE datasets with both transformations
        pure_dirs = [d for d in subdirs 
                    if d.startswith('PURE_') 
                    and 'DiffNormalized' in d 
                    and 'Standardized' in d]
        
        if pure_dirs:
            exp_data_name = pure_dirs[0]  # Use first match
            print(f"Found PURE dataset: {exp_data_name}")
        else:
            print("Available directories:")
            for d in sorted(subdirs)[:10]:
                print(f"  - {d}")
            raise ValueError("Could not auto-detect PURE dataset with DiffNormalized+Standardized. "
                            "Please set exp_data_name manually.")
    else:
        raise ValueError(f"Base cached path does not exist: {base_cached_path}")

print(f"Using EXP_DATA_NAME: {exp_data_name}")

# Set output directory in test_runs/exp folder
# Find project root (go up from analysis/ to rPPG-Toolbox/)
try:
    # If running as script
    project_root = Path(__file__).parent.parent.absolute()
except NameError:
    # If running in Jupyter notebook, try to find project root
    cwd = Path.cwd()
    if 'analysis' in str(cwd):
        project_root = cwd.parent
    else:
        project_root = Path('/home/yj167/Desktop/rPPG-Toolbox')

# Output directory: scripts/test_runs/exp/{exp_data_name}/standardized_frames
output_dir = project_root / "scripts" / "test_runs" / "exp" / exp_data_name / "standardized_frames"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# Build file list cache
print("\nBuilding file list cache...")
file_list_cache = build_file_list_cache(base_cached_path, exp_data_name)

if not file_list_cache:
    raise ValueError("No cached files found! Make sure the dataset is preprocessed.")

print(f"Found {len(file_list_cache)} cached files")

# Get unique video IDs
video_ids = set()
for (vid_id, chunk_idx) in file_list_cache.keys():
    video_ids.add(vid_id)

video_ids = sorted(list(video_ids))
print(f"Found {len(video_ids)} unique videos")

if len(video_ids) < num_samples:
    print(f"Warning: Only {len(video_ids)} videos available, but {num_samples} samples requested.")
    print(f"Will use all available videos.")
    num_samples = len(video_ids)

# Randomly sample videos
sampled_video_ids = random.sample(video_ids, num_samples)
print(f"\nSampling {num_samples} videos: {sampled_video_ids}")

# For each video, randomly select one chunk
samples = []
for video_id in sampled_video_ids:
    # Get all chunks for this video
    video_chunks = [(vid, ch) for (vid, ch) in file_list_cache.keys() 
                    if vid == video_id]
    
    if not video_chunks:
        continue
    
    # Randomly select one chunk
    selected_chunk = random.choice(video_chunks)
    samples.append(selected_chunk)

print(f"\nSelected {len(samples)} samples (one chunk per video)")

# Visualize each sample
print("\nVisualizing frames...")
for idx, (video_id, chunk_idx) in enumerate(tqdm(samples, desc="Processing")):
    # Find video file
    video_file = find_video_cache_file(base_cached_path, str(video_id), 
                                        int(chunk_idx), file_list_cache, exp_data_name)
    
    if not video_file or not os.path.exists(video_file):
        print(f"Warning: Could not find video file for {video_id}_chunk{chunk_idx}")
        continue
    
    try:
        # Load video chunk
        video_chunk = np.load(video_file)  # Shape: (T, H, W, C)
        
        # Select middle frame
        num_frames = video_chunk.shape[0]
        frame_idx = num_frames // 2
        
        # Create output filename (sequential: image1.png, image2.png, ...)
        output_filename = f"image{idx + 1}.png"
        output_path = output_dir / output_filename
        
        # Visualize
        visualize_frame_comparison(video_chunk, video_id, chunk_idx, frame_idx, output_path)
        
    except Exception as e:
        print(f"Error processing {video_id}_chunk{chunk_idx}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n✓ Visualization complete!")
print(f"Saved {len(samples)} images to: {output_dir}")

# %%
