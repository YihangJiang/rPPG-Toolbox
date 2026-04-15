# %%
# %reload_ext autoreload
# %autoreload 2
"""
MediaPipe Tasks API (FaceLandmarker) version of roi_landmarks_visualize.py.

Run this file cell-by-cell in Jupyter / VS Code Interactive — do not use
``python roi_landmarks_visualize_new.py``; there is no CLI.

Requires: face_landmarker.task (or another FaceLandmarker .task model) on disk.
"""

# %%
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# %%
# Set BEFORE any mediapipe import (avoids EGL/GPU kernel crashes on some setups).
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

# %%
# Region definitions (MediaPipe face landmark indices). Same idea as video_utils.py.
REGIONS = {
    "left_eye_region": {"keypoints": [276, 285, 464, 446], "warped_corners": [1, 2, 3, 0]},
    "right_malar": {
        "keypoints": [355, 429, 358, 423, 425, 411, 376, 352, 345, 346, 347, 329],
        "warped_corners": [0, 1, 6, 8],
    },
}


@dataclass(frozen=True)
class FaceLandmarks:
    pts_xy: np.ndarray

    @staticmethod
    def from_normalized(landmarks: Sequence[object], w: int, h: int) -> "FaceLandmarks":
        pts = np.zeros((len(landmarks), 2), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            pts[i, 0] = float(lm.x) * w
            pts[i, 1] = float(lm.y) * h
        return FaceLandmarks(pts_xy=pts)


def _ensure_region(region: str) -> dict:
    if region not in REGIONS:
        raise ValueError(f"Unknown region '{region}'. Available: {sorted(REGIONS.keys())}")
    spec = REGIONS[region]
    if not spec.get("warped_corners"):
        raise ValueError(f"Region '{region}' needs warped_corners in REGIONS.")
    return spec


def warp_region(
    frame_bgr: np.ndarray,
    face: FaceLandmarks,
    region: str,
    out_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    spec = _ensure_region(region)
    keypoints: List[int] = list(spec["keypoints"])
    corners: List[int] = list(spec["warped_corners"])
    if max(keypoints) >= face.pts_xy.shape[0]:
        raise ValueError(
            f"Landmarks count {face.pts_xy.shape[0]} < needed index {max(keypoints)}."
        )
    src_pts = face.pts_xy[np.array(keypoints, dtype=np.int32)]
    src_quad = src_pts[np.array(corners, dtype=np.int32)].astype(np.float32)
    out_w, out_h = int(out_size[0]), int(out_size[1])
    dst_quad = np.array(
        [[0, 0], [0, out_h - 1], [out_w - 1, out_h - 1], [out_w - 1, 0]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    roi = cv2.warpPerspective(frame_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return roi, src_quad


def draw_landmarks_basic(
    frame_bgr: np.ndarray, face: FaceLandmarks, idxs: Optional[Iterable[int]] = None
) -> None:
    h, w = frame_bgr.shape[:2]
    pts = face.pts_xy
    if idxs is None:
        idxs = range(min(len(pts), 468))
    for i in idxs:
        x, y = pts[int(i)]
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(frame_bgr, (xi, yi), 1, (0, 255, 0), -1)


def make_face_landmarker(model_path: str, num_faces: int = 2):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model: {model_path}\n"
            "Download a FaceLandmarker .task file and set model_path in the paths cell."
        )
    base_options = python.BaseOptions(
        model_asset_path=model_path,
        delegate=python.BaseOptions.Delegate.CPU,
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=num_faces,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def detect_first_face(mp, landmarker, frame_bgr: np.ndarray) -> Optional[FaceLandmarks]:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    return FaceLandmarks.from_normalized(result.face_landmarks[0], w=w, h=h)


def annotate_video_with_rois(
    input_video_path: str,
    output_video_path: str,
    mp,
    face_landmarker,
    region_name: str,
    roi_size: Tuple[int, int],
    draw_all_landmarks: bool = False,
    max_frames: int = -1,
) -> None:
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        raise ValueError("Could not read video width/height.")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise ValueError(f"Cannot open writer: {output_video_path}")
    spec = _ensure_region(region_name)
    region_keypoints = spec["keypoints"]
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if max_frames > 0 and frame_i > max_frames:
            break
        face = detect_first_face(mp, face_landmarker, frame)
        if face is not None:
            if draw_all_landmarks:
                draw_landmarks_basic(frame, face, idxs=None)
            else:
                draw_landmarks_basic(frame, face, idxs=region_keypoints)
            try:
                roi, quad = warp_region(frame, face, region=region_name, out_size=roi_size)
                quad_int = np.round(quad).astype(int)
                cv2.polylines(
                    frame, [quad_int.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2
                )
                preview = cv2.resize(roi, (roi_size[0] // 2, roi_size[1] // 2))
                ph, pw = preview.shape[:2]
                frame[0:ph, 0:pw] = preview
            except Exception:
                pass
        writer.write(frame)
    writer.release()
    cap.release()

# %%
import matplotlib.pyplot as plt

# %%
# BUAA Sub_08 lux 25.1 — edit paths as needed
input_video_path = "/mnt/nvme2/rppg_data/BUAA/Sub_08/lux 25.1/lux25.1_LZY.avi"
output_video_path = "./plot1_new.avi"
model_path = "face_landmarker.task"
region_name = "left_eye_region"  # or "right_malar"
roi_size = (320, 320)

# %%
vidcap = cv2.VideoCapture(input_video_path)
success, image = vidcap.read()
if not success:
    print("Cannot read the video")
else:
    print(f"Loaded frame from {input_video_path}")

# %%
if success:
    img = image[:, :1200, :]
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
else:
    print("Skip: no frame loaded.")

# %%
mp, landmarker = make_face_landmarker(model_path=model_path, num_faces=2)

# %%
face = None
if success:
    face = detect_first_face(mp, landmarker, image)
if face is None:
    print("No face in first frame (or no frame loaded).")
else:
    print("Landmark count:", face.pts_xy.shape[0])

# %%
if not success:
    print("Skip: no frame loaded.")
elif face is None:
    print("Skip: no face.")
else:
    annotated = image.copy()
    draw_landmarks_basic(annotated, face, idxs=REGIONS[region_name]["keypoints"])
    try:
        roi, quad = warp_region(annotated, face, region=region_name, out_size=roi_size)
        quad_int = np.round(quad).astype(int)
        cv2.polylines(annotated, [quad_int.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Annotated frame")
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title(f"Warped ROI: {region_name}")
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"ROI warp failed: {e}")

# %%
# Full-video annotation (same pattern as old annotate_video_with_rois call)
# Run only after landmarker cell above has run.
# annotate_video_with_rois(
#     input_video_path, output_video_path, mp, landmarker, region_name, roi_size
# )
