# %%
%reload_ext autoreload
%autoreload 2

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import mediapipe as mp
from video_utils import *
# %%

vidcap = cv2.VideoCapture('/hpc/group/dunnlab/yj167/DATASET_1/s1/vid_s1_T2.avi')
success, image = vidcap.read()
# %%
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

facial_areas = {
    'Contours': mp_face_mesh.FACEMESH_CONTOURS
    , 'Lips': mp_face_mesh.FACEMESH_LIPS
    , 'Face_oval': mp_face_mesh.FACEMESH_FACE_OVAL
    , 'Left_eye': mp_face_mesh.FACEMESH_LEFT_EYE
    , 'Left_eye_brow': mp_face_mesh.FACEMESH_LEFT_EYEBROW
    , 'Right_eye': mp_face_mesh.FACEMESH_RIGHT_EYE
    , 'Right_eye_brow': mp_face_mesh.FACEMESH_RIGHT_EYEBROW
    , 'Tesselation': mp_face_mesh.FACEMESH_TESSELATION
}

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5)

results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
len(results.multi_face_landmarks[0].landmark)

# %%
