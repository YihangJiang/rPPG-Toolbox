# %%
# %reload_ext autoreload
# %autoreload 2
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import mediapipe as mp
from video_utils import *

src_root = "/work/yj167/DATASET_1"
dst_root = "/work/yj167/DATASET_1IN"

list_phys1, list_phys2 = get_ubfc_paths(src_root, dst_root)

# %%
list_phys1 = [
    f for f in list_phys1
    if int(f.split('/')[4][1:]) < 30
]

list_phys2 = [
    f for f in list_phys2
    if int(f.split('/')[4][1:]) < 30

]

# %%
for i in range(len(list_phys1)):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5)
    input_video_path, output_video_path = list_phys1[i], list_phys2[i]
    print(list_phys1[i], list_phys2[i])
    annotate_video_with_rois(input_video_path, output_video_path, face_mesh, "infraorbital", (320,320))

# %%
