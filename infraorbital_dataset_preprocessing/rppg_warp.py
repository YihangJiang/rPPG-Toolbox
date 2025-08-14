import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from video_utils import *
area_names = ['Right_eye', 'Left_eye']

src_root = "/hpc/group/dunnlab/rppg_data/data/DATASET_2"
dst_root = "/work/yj167/DATASET_2ROI"

import mediapipe as mp


list_rppg1, list_rppg2 = get_ubfc_paths(src_root, dst_root)
for i in range(len(list_rppg1)):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5)
    input_video_path, output_video_path = list_rppg1[i], list_rppg2[i]
    print(list_rppg1[i], list_rppg2[i])
    annotate_video_with_rois(input_video_path, output_video_path, face_mesh, "right malar", (320,320))
