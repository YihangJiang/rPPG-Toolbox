#!/usr/bin/env python3
"""
Demo script for heart rate inference from a video file.

Edit the hardcoded paths at the top of main() function to use your video and model.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import cv2
from scipy.signal import butter, filtfilt
import scipy.signal

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()  # Go up one level from demo/ to project root
sys.path.insert(0, str(project_root))

from config import get_config, _C
from dataset.data_loader.BaseLoader import BaseLoader
from neural_methods.model.TS_CAN import TSCAN
from evaluation.post_process import _calculate_fft_hr, _detrend


def read_video(video_path):
    """Read video file and return frames as numpy array (T, H, W, 3) in RGB format."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    frames = np.array(frames)
    print(f"Loaded video: {len(frames)} frames, shape: {frames.shape}, FPS: {fps:.2f}")
    return frames, fps


def preprocess_video(frames, fps, config_preprocess, base_loader):
    """Preprocess video frames using BaseLoader preprocessing pipeline."""
    # Create dummy BVP signal (not used for inference, but required by preprocess)
    # Use zeros with same length as frames
    dummy_bvps = np.zeros(len(frames))
    
    # Use BaseLoader's preprocess method
    frames_clips, _ = base_loader.preprocess(frames, dummy_bvps, config_preprocess)
    
    print(f"Preprocessed into {len(frames_clips)} clip(s)")
    print(f"Clip shape: {frames_clips[0].shape}")
    
    return frames_clips, fps


def load_model(model_path, config):
    """Load TSCAN model with appropriate configuration."""
    device = torch.device(config.DEVICE if hasattr(config, 'DEVICE') else 'cuda:0')
    
    # Determine input channels from config
    num_rgb_channels = 1 if hasattr(config.TEST.DATA.PREPROCESS, 'COLOR_CHANNEL') and config.TEST.DATA.PREPROCESS.COLOR_CHANNEL else 3
    num_transformations = len(config.TEST.DATA.PREPROCESS.DATA_TYPE)
    total_channels = num_rgb_channels * num_transformations
    in_channels = total_channels // 2  # Each branch gets half the total channels
    
    frame_depth = config.MODEL.TSCAN.FRAME_DEPTH if hasattr(config.MODEL, 'TSCAN') else 10
    img_size = config.TEST.DATA.PREPROCESS.RESIZE.H
    
    print(f"Initializing TSCAN model:")
    print(f"  - Input channels per branch: {in_channels}")
    print(f"  - Frame depth: {frame_depth}")
    print(f"  - Image size: {img_size}")
    
    model = TSCAN(in_channels=in_channels, frame_depth=frame_depth, img_size=img_size)
    
    # Load model weights
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel wrapper (remove 'module.' prefix if present)
    if any(key.startswith('module.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model, device


def predict_ppg(model, frames_clips, device, chunk_length, frame_depth, num_gpu=1):
    """Run inference on preprocessed video clips."""
    base_len = num_gpu * frame_depth
    all_predictions = []
    
    print("Running inference...")
    with torch.no_grad():
        for clip_idx, clip in enumerate(frames_clips):
            # Convert to torch tensor and handle format
            # Clip is in format (chunk_length, H, W, C)
            # Need to convert to (N*D, C, H, W) for model input
            clip = np.float32(clip)
            T, H, W, C = clip.shape
            
            # Transpose to (T, C, H, W)
            clip = np.transpose(clip, (0, 3, 1, 2))
            
            # Pad or truncate to be divisible by base_len
            num_frames = len(clip)
            if num_frames < base_len:
                # Pad with last frame
                padding = np.repeat(clip[-1:], base_len - num_frames, axis=0)
                clip = np.concatenate([clip, padding], axis=0)
            
            # Truncate to be divisible by base_len
            clip = clip[:(len(clip) // base_len) * base_len]
            
            # Convert to batch format: (N*D, C, H, W) where N*D = num_frames
            clip_tensor = torch.from_numpy(clip).unsqueeze(0)  # Add batch dimension: (1, T, C, H, W)
            clip_tensor = clip_tensor.view(-1, C, H, W)  # (T, C, H, W)
            clip_tensor = clip_tensor.to(device)
            
            # Run inference
            pred_ppg = model(clip_tensor)
            pred_ppg = pred_ppg.cpu().numpy().squeeze()
            
            # Only keep predictions for original frames (remove padding)
            if len(pred_ppg) > num_frames:
                pred_ppg = pred_ppg[:num_frames]
            
            all_predictions.append(pred_ppg)
            print(f"  Clip {clip_idx + 1}/{len(frames_clips)}: Predicted {len(pred_ppg)} PPG values")
    
    # Concatenate all predictions
    if len(all_predictions) > 1:
        ppg_signal = np.concatenate(all_predictions)
    else:
        ppg_signal = all_predictions[0]
    
    return ppg_signal


def calculate_heart_rate(ppg_signal, fps, diff_flag=False):
    """Calculate heart rate from PPG signal using FFT."""
    # Detrend the signal
    if diff_flag:
        # If predictions are 1st derivative, cumsum to get original signal
        ppg_signal = np.cumsum(ppg_signal)
    
    ppg_signal = _detrend(ppg_signal, 100)
    
    # Apply bandpass filter [0.75, 2.5] Hz = [45, 150] BPM
    [b, a] = butter(1, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
    ppg_signal = filtfilt(b, a, np.double(ppg_signal))
    
    # Calculate HR using FFT
    hr = _calculate_fft_hr(ppg_signal, fs=fps, low_pass=0.75, high_pass=2.5)
    
    return hr, ppg_signal


def main():
    # ============================================================================
    # HARDCODED PATHS - Edit these to use your video and model
    # ============================================================================
    video_path = "input.mp4"  # Path to input video file
    model_path = "./final_model_release/PURE_TSCAN.pth"  # Path to trained model .pth file
    config_path = None  # Path to config YAML file (optional, set to None to use defaults)
    device = "cuda:0"  # Device to use: "cuda:0" or "cpu"
    # ============================================================================
    
    # Load config
    if config_path:
        from types import SimpleNamespace
        config_args = SimpleNamespace()
        config_args.config_file = config_path
        config = get_config(config_args)
    else:
        # Use default config based on the experiment config
        config = _C.clone()
        config.defrost()
        
        # Set default preprocessing parameters (matching tscan_ubfc_rppg_to_pure.yaml)
        config.TEST.DATA.PREPROCESS.DATA_TYPE = ['Raw', 'DiffNormalized']
        config.TEST.DATA.PREPROCESS.LABEL_TYPE = 'Raw'
        config.TEST.DATA.PREPROCESS.DO_CHUNK = True
        config.TEST.DATA.PREPROCESS.CHUNK_LENGTH = 180
        config.TEST.DATA.PREPROCESS.CROP_FACE.DO_CROP_FACE = True
        config.TEST.DATA.PREPROCESS.CROP_FACE.BACKEND = 'HC'
        config.TEST.DATA.PREPROCESS.CROP_FACE.USE_LARGE_FACE_BOX = True
        config.TEST.DATA.PREPROCESS.CROP_FACE.LARGE_BOX_COEF = 1.5
        config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION = True
        config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY = 30
        config.TEST.DATA.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX = True
        config.TEST.DATA.PREPROCESS.RESIZE.H = 72
        config.TEST.DATA.PREPROCESS.RESIZE.W = 72
        config.TEST.DATA.PREPROCESS.DATA_AUG = ['None']
        config.MODEL.TSCAN.FRAME_DEPTH = 10
        config.DEVICE = device
        config.freeze()
    
    # Read video first to get FPS
    print("=" * 60)
    print("Reading video...")
    print("=" * 60)
    frames, fps = read_video(video_path)
    
    # Initialize BaseLoader for preprocessing (we won't use it fully, just for preprocessing methods)
    class DummyLoader(BaseLoader):
        def get_raw_data(self, raw_data_path):
            return []
        def split_raw_data(self, data_dirs, begin, end):
            return []
    
    # Set DO_PREPROCESS to False to avoid loading cached data
    config.defrost()
    config.TEST.DATA.DO_PREPROCESS = False
    config.TEST.DATA.FS = fps  # Set FPS from video
    # Set dummy paths to avoid errors
    if not hasattr(config.TEST.DATA, 'CACHED_PATH') or not config.TEST.DATA.CACHED_PATH:
        config.TEST.DATA.CACHED_PATH = '/tmp/demo_cache'
    if not hasattr(config.TEST.DATA, 'FILE_LIST_PATH') or not config.TEST.DATA.FILE_LIST_PATH:
        config.TEST.DATA.FILE_LIST_PATH = '/tmp/demo_filelist.csv'
    # Set required attributes
    if not hasattr(config.TEST.DATA, 'DATA_FORMAT'):
        config.TEST.DATA.DATA_FORMAT = 'NDHWC'
    if not hasattr(config.TEST.DATA, 'BEGIN'):
        config.TEST.DATA.BEGIN = 0.0
    if not hasattr(config.TEST.DATA, 'END'):
        config.TEST.DATA.END = 1.0
    if not hasattr(config.TEST.DATA, 'EDA_PATH'):
        config.TEST.DATA.EDA_PATH = ''
    config.freeze()
    
    base_loader = DummyLoader(
        dataset_name="demo",
        raw_data_path="",
        config_data=config.TEST.DATA
    )
    
    # Preprocess video
    print("\n" + "=" * 60)
    print("Preprocessing video...")
    print("=" * 60)
    frames_clips, fps = preprocess_video(frames, fps, config.TEST.DATA.PREPROCESS, base_loader)
    
    # Load model
    print("\n" + "=" * 60)
    print("Loading model...")
    print("=" * 60)
    model, device = load_model(model_path, config)
    
    # Run inference
    print("\n" + "=" * 60)
    print("Running inference...")
    print("=" * 60)
    frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
    chunk_length = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
    ppg_signal = predict_ppg(model, frames_clips, device, chunk_length, frame_depth)
    
    # Calculate heart rate
    print("\n" + "=" * 60)
    print("Calculating heart rate...")
    print("=" * 60)
    
    # Check if label type is DiffNormalized (means predictions are also diff normalized)
    diff_flag = config.TEST.DATA.PREPROCESS.LABEL_TYPE == 'DiffNormalized'
    hr, filtered_ppg = calculate_heart_rate(ppg_signal, fps, diff_flag=diff_flag)
    
    # Output results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Input video: {video_path}")
    print(f"Video FPS: {fps:.2f}")
    print(f"Video duration: {len(frames) / fps:.2f} seconds")
    print(f"Total frames: {len(frames)}")
    print(f"Predicted PPG signal length: {len(ppg_signal)}")
    print(f"\nEstimated Heart Rate: {hr:.2f} BPM")
    print("=" * 60)
    
    return hr


if __name__ == "__main__":
    main()

