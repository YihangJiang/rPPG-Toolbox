# %%
"""
Preprocess UBFC-style videos in DATASET_2 to spatial-temporal maps (stmaps).
- Face detection (Haar), then 5x5 grid on face crop → 25 blocks; mean per block → stmap (25, num_frames, 3).
- Default channels (BGR); no explicit RGB/YUV.
- Stmap normalized to 0-255; PPG signal (ground_truth.txt) resampled to video length and normalized to 0-1.
- Runnable as script or in Jupyter (run cells in order).
"""

import os
import numpy as np
import cv2
from video_utils import get_ubfc_paths

# Project root for Haar cascade (script lives in infraorbital_dataset_preprocessing/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
CASCADE_PATH = os.path.join(_PROJECT_ROOT, "dataset", "haarcascade_frontalface_default.xml")

# Default paths (override in next cell or when calling main())
SRC_ROOT = "/mnt/nvme2/rppg_data/DATASET_2"
DST_ROOT = "/mnt/nvme2/rppg_data/STMap/UBFC"
GRID_H, GRID_W = 5, 5  # 5x5 -> 25 blocks

# %%
# Optional: override paths / grid for your run
# SRC_ROOT = "/mnt/mvme2/rppg_data/DATASET_2"
# DST_ROOT = "/mnt/mvme2/rppg_data/STMap/UBFC"
# GRID_H, GRID_W = 5, 5

# %%
def chunkify(img, block_height=5, block_width=5):
    """Split image into a grid of blocks; return list of blocks (each block = small image patch)."""
    h, w = img.shape[:2]
    y_len = max(1, h // block_height)
    x_len = max(1, w // block_width)
    blocks = []
    for i in range(block_height):
        start_y = i * y_len
        end_y = (h if i == block_height - 1 else (i + 1) * y_len)
        for j in range(block_width):
            start_x = j * x_len
            end_x = (w if j == block_width - 1 else (j + 1) * x_len)
            blocks.append(img[start_y:end_y, start_x:end_x])
    return blocks


def get_face_detector():
    """Return Haar cascade classifier for face detection."""
    if not os.path.isfile(CASCADE_PATH):
        raise FileNotFoundError(f"Haar cascade not found: {CASCADE_PATH}")
    return cv2.CascadeClassifier(CASCADE_PATH)


def detect_face_one_frame(frame, detector):
    """Return [x, y, w, h] for largest face, or None to use full frame."""
    zones = detector.detectMultiScale(frame)
    if len(zones) == 0:
        return None
    if len(zones) >= 2:
        idx = np.argmax(zones[:, 2])
        return zones[idx].tolist()
    return zones[0].tolist()


def video_to_stmap_25(video_path, detector, grid_h=5, grid_w=5):
    """
    Load video, detect face on first frame, crop all frames to that box,
    then for each frame split crop into grid_h x grid_w blocks and take mean per block (default BGR).
    Returns:
        stmap: (25, num_frames, 3) float, unnormalized (raw means).
        num_frames: int.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0
    num_blocks = grid_h * grid_w
    frames_means = []  # list of (25, 3)
    face_box = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if face_box is None:
            face_box = detect_face_one_frame(frame, detector)
        if face_box is not None:
            x, y, w, h = face_box
            frame = frame[
                max(0, y) : min(y + h, frame.shape[0]),
                max(0, x) : min(x + w, frame.shape[1]),
            ]
        if frame.size == 0:
            continue
        blocks = chunkify(frame, grid_h, grid_w)
        if len(blocks) != num_blocks:
            continue
        means = np.array([np.mean(b, axis=(0, 1)) for b in blocks], dtype=np.float32)  # (25, 3)
        frames_means.append(means)
    cap.release()
    if not frames_means:
        return None, 0
    stmap = np.stack(frames_means, axis=1)  # (25, T, 3)
    return stmap, stmap.shape[1]


def normalize_stmap_0_255(stmap):
    """MinMax per (block, channel) over time to 0-255, return uint8 (25, T, 3)."""
    out = np.zeros_like(stmap, dtype=np.float64)
    for b in range(stmap.shape[0]):
        for c in range(stmap.shape[2]):
            x = stmap[b, :, c]
            lo, hi = x.min(), x.max()
            if hi > lo:
                out[b, :, c] = (x - lo) / (hi - lo) * 255.0
            else:
                out[b, :, c] = 0.0
    return np.clip(out, 0, 255).astype(np.uint8)


def read_ubfc_ground_truth(video_path):
    """UBFC-rPPG: ground_truth.txt in same dir as video (space-separated floats, one line)."""
    dir_path = os.path.dirname(video_path)
    gt_path = os.path.join(dir_path, "ground_truth.txt")
    if not os.path.isfile(gt_path):
        return None
    with open(gt_path, "r") as f:
        line = f.read().strip().split("\n")[0]
    values = [float(x) for x in line.split()]
    return np.asarray(values, dtype=np.float64)


def resample_signal(signal, target_length):
    """Resample 1D signal to target_length (same sampling as video frames)."""
    if len(signal) == 0:
        return np.zeros(target_length, dtype=np.float32)
    return np.interp(
        np.linspace(0, len(signal) - 1, target_length),
        np.arange(len(signal)),
        signal,
    ).astype(np.float32)


def normalize_signal_0_1(signal):
    """MinMax to [0, 1], float32."""
    x = np.asarray(signal, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi > lo:
        x = (x - lo) / (hi - lo)
    else:
        x = np.zeros_like(x)
    return x.astype(np.float32)


def process_one_video(input_path, detector, grid_h=5, grid_w=5):
    """
    Generate stmap (25, T, 3) uint8 and signal (T,) float32 for one video.
    Returns (stmap, signal) or (None, None) on failure.
    """
    stmap_raw, T = video_to_stmap_25(input_path, detector, grid_h, grid_w)
    if stmap_raw is None or T == 0:
        return None, None
    stmap_u8 = normalize_stmap_0_255(stmap_raw)
    bvp = read_ubfc_ground_truth(input_path)
    if bvp is not None:
        bvp_resampled = resample_signal(bvp, T)
        signal = normalize_signal_0_1(bvp_resampled)
    else:
        signal = np.zeros(T, dtype=np.float32)
    return stmap_u8, signal


# %%
def run_all(src_root=None, dst_root=None, grid_h=5, grid_w=5):
    """Discover all UBFC .avi under src_root; write stmap as .png and .npy, signal as .npy under dst_root."""
    src_root = src_root or SRC_ROOT
    dst_root = dst_root or DST_ROOT
    list_src, list_dst = get_ubfc_paths(src_root, dst_root)
    if not list_src:
        print(f"No .avi videos found under {src_root}")
        return
    detector = get_face_detector()
    print(f"Found {len(list_src)} videos. Writing to {dst_root} (stmap 0-255 PNG+npy, signal 0-1 npy).")
    for i, (input_path, output_path) in enumerate(zip(list_src, list_dst)):
        base = os.path.splitext(output_path)[0]
        stmap_png_path = base + "_stmap.png"
        stmap_npy_path = base + "_stmap.npy"
        signal_path = base + "_signal.npy"
        os.makedirs(os.path.dirname(stmap_png_path), exist_ok=True)
        stmap, signal = process_one_video(input_path, detector, grid_h, grid_w)
        if stmap is None:
            print(f"  Skip: {input_path}")
            continue
        # stmap (25, T, 3) uint8 -> PNG (height 25, width T)
        cv2.imwrite(stmap_png_path, stmap)
        np.save(stmap_npy_path, stmap)
        np.save(signal_path, signal)
        print(f"  [{i+1}/{len(list_src)}] {os.path.relpath(input_path, src_root)} -> stmap {stmap.shape} .png+.npy, signal {signal.shape}")
    print("Done.")


# %%
# Run preprocessing (execute this cell to process all videos)
run_all(SRC_ROOT, DST_ROOT, GRID_H, GRID_W)


# %%
for i,j,k in os.walk(SRC_ROOT):
    print(i,j,k)
# %%
