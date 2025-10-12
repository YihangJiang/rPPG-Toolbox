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
area_names = ['Right_eye', 'Left_eye']


# %%
input_video_path, output_video_path = '/work/yj167/DATASET_1/s1/vid_s1_T1.avi', './plot1.avi'
vidcap = cv2.VideoCapture(input_video_path)
success, image = vidcap.read()
if not success:
    print("Cannot read the video")
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
# landmarks = results.multi_face_landmarks[0]
# for facial_area in facial_areas.keys():
#     facial_area_obj = facial_areas[facial_area]
#     plot_landmark(annotated_image, facial_area, results, pt_min, pt_max, False)
#     cv2.circle(annotated_image, pt_min, 10, (255, 0, 0), -1)
#     cv2.circle(annotated_image, pt_max, 10, (255, 0, 0), -1)
plot_rois(results, image, "infraorbital")
# %%
# masked_image, _ = plot_semi(annotated_image, pt_min, pt_max, False)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5)
annotate_video_with_rois(input_video_path, output_video_path, face_mesh, "infraorbital", (320,320))

# %%
results = face_detection(image)
annotated_image = image.copy()
seq_num_list_1 = get_seq_num_facial_areas(facial_areas, area_names[0])
seq_num_list_2 = get_seq_num_facial_areas(facial_areas, area_names[1])
time4 = time.time()
pt_min, pt_max = locate_eye_corner(results, seq_num_list_1, annotated_image)
pt_min_2, pt_max_2 = locate_eye_corner(results, seq_num_list_2, annotated_image)
time5 = time.time()
# plot_landmark(annotated_image, area_name, results, pt_min, pt_max, True)
masked_image, extracted_pixels = plot_semi(annotated_image, [pt_min, pt_min_2], [pt_max, pt_max_2], False)
# %%
src_root = "/work/yj167/DATASET_1"
dst_root = "/work/yj167/DATASET_ROI"

list_phys1, list_phys2 = get_ubfc_paths(src_root, dst_root)
# %%
src_root = "/hpc/group/dunnlab/rppg_data/data/DATASET_2"
dst_root = "/work/yj167/DATASET_2ROI"

list_rppg1, list_rppg2 = get_ubfc_paths(src_root, dst_root)


# %%
for i in range(len(list_1)):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=2,
        min_detection_confidence=0.5)
    input_video_path, output_video_path = list_1[i], list_2[i]
    print(list_1[i], list_2[i])
    annotate_video_with_rois(input_video_path, output_video_path, face_mesh, "right malar", (320,320))


# %%
