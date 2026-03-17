# %%
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from video_utils import *
import mediapipe as mp

src_root = "/mnt/nvme2/rppg_data/BUAA"
dst_root = "/mnt/nvme2/rppg_data/BUAA_RM"

list_buaa1, list_buaa2 = get_ubfc_paths(src_root, dst_root)

# %%
for i in range(len(list_buaa1)):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5)
    input_video_path, output_video_path = list_buaa1[i], list_buaa2[i]
    print(list_buaa1[i], list_buaa2[i])
    annotate_video_with_rois(input_video_path, output_video_path, face_mesh, "left malar", (320,320))


# %%

