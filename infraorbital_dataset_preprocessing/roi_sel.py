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

vidcap = cv2.VideoCapture('./test.mov')
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

# %%
img = image[:,:800,]
plt.imshow(img)

results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
len(results.multi_face_landmarks[0].landmark)

# %%
# Calculate landmarks number
len(results.multi_face_landmarks[0].landmark)
annotated_image = image.copy()
for face_landmarks in results.multi_face_landmarks:
    print(face_landmarks)
    mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_iris_connections_style())
resize_and_show(annotated_image)
# %%
# Get the contour of specific facial areas
selected_facial_area = facial_areas['Right_eye']
seq_num_list = np.array([])
for i in selected_facial_area:
    seq_num_list = np.append(seq_num_list, i[0])
    seq_num_list = np.append(seq_num_list, i[1])

seq_num_list = np.unique(seq_num_list.astype(int))
print(np.unique(seq_num_list))
# %%
# Get the corner of eyes
pt_min, pt_max = locate_eye_corner(results, seq_num_list, annotated_image)

# %%
landmarks = results.multi_face_landmarks[0]
for facial_area in facial_areas.keys():
    facial_area_obj = facial_areas[facial_area]
    plot_landmark(annotated_image, facial_area, results, pt_min, pt_max, False)
    cv2.circle(annotated_image, pt_min, 10, (255, 0, 0), -1)
    cv2.circle(annotated_image, pt_max, 10, (255, 0, 0), -1)
# %%
masked_image, _ = plot_semi(annotated_image, pt_min, pt_max, False)
plot_rois(results, image)


# %%



# %%
