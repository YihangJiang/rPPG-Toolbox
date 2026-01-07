"""
Feature extraction functions for signal and video data.

This module provides functions to extract high-dimensional features from:
1. PPG/BVP signals - frequency domain, statistical, temporal features
2. Video chunks - optical flow, color statistics, motion, texture features
"""

import numpy as np
import cv2
from scipy import stats
from scipy.fft import fft, fftfreq


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
            features['dominant_freq'] = pos_freqs[np.argmax(pos_fft[hr_mask])]
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
        very_low = (pos_freqs >= 0.04) & (pos_freqs < 0.15)  # Very low frequency
        low = (pos_freqs >= 0.15) & (pos_freqs < 0.4)  # Low frequency
        high = (pos_freqs >= 0.4) & (pos_freqs < 0.6)  # High frequency
        hr_band = (pos_freqs >= 0.5) & (pos_freqs <= 4.0)  # Heart rate band
        
        features['power_vlf'] = np.sum(pos_fft[very_low]) if np.any(very_low) else 0
        features['power_lf'] = np.sum(pos_fft[low]) if np.any(low) else 0
        features['power_hf'] = np.sum(pos_fft[high]) if np.any(high) else 0
        features['power_hr'] = np.sum(pos_fft[hr_band]) if np.any(hr_band) else 0
        features['lf_hf_ratio'] = features['power_lf'] / (features['power_hf'] + 1e-10)
    else:
        # Default values if signal is too short
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


def extract_video_features(video_chunk):
    """Extract high-dimensional features from video chunk.
    
    Args:
        video_chunk: Video array of shape (T, H, W, C)
        
    Returns:
        dict: Dictionary of extracted features
    """
    # Video chunks are expected to be in (T, H, W, C) format
    # if len(video_chunk.shape) != 4:
    #     raise ValueError(f"Expected 4D video array, got shape: {video_chunk.shape}")
    T, H, W, C = video_chunk.shape
    
    features = {}
    
    # Color statistics per channel
    for ch_idx in range(C):
        ch_data = video_chunk[:, :, :, ch_idx]
        features[f'ch{ch_idx}_mean'] = np.mean(ch_data)
        features[f'ch{ch_idx}_std'] = np.std(ch_data)
        features[f'ch{ch_idx}_min'] = np.min(ch_data)
        features[f'ch{ch_idx}_max'] = np.max(ch_data)
        features[f'ch{ch_idx}_median'] = np.median(ch_data)
    
    # Overall brightness and contrast
    if C >= 3:  # RGB
        gray = np.mean(video_chunk[:, :, :, :3], axis=3)  # Convert to grayscale
        features['brightness_mean'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['contrast'] = np.std(gray)
    else:
        gray = video_chunk[:, :, :, 0]
        features['brightness_mean'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['contrast'] = np.std(gray)
    
    # Temporal variation (motion)
    if T > 1:
        frame_diffs = np.diff(gray, axis=0)
        features['motion_mean'] = np.mean(np.abs(frame_diffs))
        features['motion_std'] = np.std(frame_diffs)
        features['motion_max'] = np.max(np.abs(frame_diffs))
        
        # Optical flow approximation (simplified)
        flow_magnitude = []
        for t in range(T - 1):
            frame1 = gray[t].astype(np.float32)
            frame2 = gray[t + 1].astype(np.float32)
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            flow_magnitude.append(np.mean(flow_mag))
        
        if flow_magnitude:
            features['optical_flow_mean'] = np.mean(flow_magnitude)
            features['optical_flow_std'] = np.std(flow_magnitude)
            features['optical_flow_max'] = np.max(flow_magnitude)
        else:
            features['optical_flow_mean'] = 0
            features['optical_flow_std'] = 0
            features['optical_flow_max'] = 0
    else:
        features['motion_mean'] = 0
        features['motion_std'] = 0
        features['motion_max'] = 0
        features['optical_flow_mean'] = 0
        features['optical_flow_std'] = 0
        features['optical_flow_max'] = 0
    
    # Texture features (using Laplacian variance as texture measure)
    texture_scores = []
    for t in range(T):
        frame = gray[t].astype(np.uint8)
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        texture_scores.append(np.var(laplacian))
    
    features['texture_mean'] = np.mean(texture_scores)
    features['texture_std'] = np.std(texture_scores)
    features['texture_max'] = np.max(texture_scores)
    
    # Spatial gradient features
    gradient_magnitudes = []
    for t in range(T):
        frame = gray[t].astype(np.uint8)
        grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitudes.append(np.mean(grad_mag))
    
    features['gradient_mean'] = np.mean(gradient_magnitudes)
    features['gradient_std'] = np.std(gradient_magnitudes)
    
    # Color distribution (histogram statistics)
    if C >= 3:
        for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
            ch_data = video_chunk[:, :, :, ch_idx].flatten()
            hist, _ = np.histogram(ch_data, bins=32, range=(0, 255))
            hist = hist / (np.sum(hist) + 1e-10)
            features[f'{ch_name}_hist_entropy'] = -np.sum(hist * np.log(hist + 1e-10))
    
    return features

