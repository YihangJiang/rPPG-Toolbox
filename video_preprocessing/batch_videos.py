# %%
%reload_ext autoreload
%autoreload 2

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import mediapipe as mp
from video_utils import *
import pandas as pd
area_name = 'Right_eye'

# %% Extract the names of the path
# Path to the main folder containing subject subfolders
main_folder = '/hpc/group/dunnlab/yj167/DATASET_2'
subject_pattern = re.compile(r'^subject\d+$')
subject_path_list = []
video_path_list = []

# Loop through each subject folder
for subject_folder in os.listdir(main_folder):
    subject_path = os.path.join(main_folder, subject_folder)
    if os.path.isdir(subject_path) and subject_pattern.match(subject_folder):  # Check if it's a directory
        subject_path_list.append(subject_path)
        for file in os.listdir(subject_path):
            if file.endswith('.avi'):  # Check if the file is a video
                video_path = os.path.join(subject_path, file)
                video_path_list.append(video_path)



# %%
columns = ["Subject", "Frame Number", "ROI Width", "ROI Height", "Area"]
meta_df = pd.DataFrame(columns=columns)
for i in range(1):
# for i in range(2):
    print(i)
    vidcap = cv2.VideoCapture(video_path_list[i])
    subject_name = re.search(r"(subject)(\d+)", subject_path_list[i]).group()
    meta_file_path = os.path.join(subject_path_list[i], f"{subject_name}_meta.csv")

    # Read each frame in a loop
    frame_count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            print("End of video or unable to read frame.")
            break

        results = face_detection(image)
        annotated_image = image.copy()
        seq_num_list = get_seq_num_facial_areas(facial_areas, area_name)
        pt_min, pt_max = locate_eye_corner(results, seq_num_list, annotated_image)


        plot_landmark(annotated_image, area_name, results, pt_min, pt_max, True)
        masked_image, extracted_pixels = plot_semi(annotated_image, pt_min, pt_max, True)

        roi_width = abs(pt_max[0] - pt_min[0])
        roi_height = abs(pt_max[1] - pt_min[1])
        area = extracted_pixels.shape[0]
        meta_df.loc[len(meta_df)] = [i, frame_count, roi_width, roi_height, area]
        
        # Process the frame (e.g., display or save it)
        print(f"Reading frame {frame_count}")
        # Uncomment to display the frame (press 'q' to quit display)
        # cv2.imshow('Frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        frame_count += 1
        if frame_count > 3:
            break




# %%
# Path to the video file
main_dir = '/hpc/group/dunnlab/yj167/rPPG-Toolbox/meta_d_2'
subject_dir = '/hpc/group/dunnlab/yj167/DATASET_1/s1/vid_s1_T2.avi'
video_name = os.path.basename(video_path).split('.')[0]
print(video_name)
# # Open video
# vidcap = cv2.VideoCapture(video_path)

# # Meta-information file path
# meta_file_path = os.path.join(output_meta_dir, f"{video_name}_meta.csv")

# # Initialize meta file
# with open(meta_file_path, mode='w', newline='') as meta_file:
#     csv_writer = csv.writer(meta_file)
#     # Write header
#     csv_writer.writerow(['Frame Number', 'ROI Width', 'ROI Height'])

#     # Read each frame in a loop
#     frame_count = 0
#     while True:
#         success, image = vidcap.read()
#         if not success:
#             print("End of video or unable to read frame.")
#             break

#         results = face_detection(image)
#         annotated_image = image.copy()
#         seq_num_list = get_seq_num_facial_areas(facial_areas, area_name)
#         pt_min, pt_max = locate_eye_corner(results, seq_num_list, annotated_image)

#         if pt_min and pt_max:
#             # Calculate ROI size
#             roi_width = abs(pt_max[0] - pt_min[0])
#             roi_height = abs(pt_max[1] - pt_min[1])

#             # Write frame data to the meta file
#             csv_writer.writerow([frame_count, roi_width, roi_height])

#             # Optional: Plot and mask the image
#             plot_landmark(annotated_image, area_name, results, pt_min, pt_max, False)
#             masked_image, extracted_pixels = plot_semi(annotated_image, pt_min, pt_max, True)

#         print(f"Processed frame {frame_count}")
#         frame_count += 1

#         # Break after 10 frames for testing
#         if frame_count > 10:
#             break

# # Release video capture
# vidcap.release()
# print(f"Meta-information saved at {meta_file_path}")





# %%
meta_df
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # Example data
# data = {
#     "Subject": [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
#     "Frame Number": [0, 1, 2, 3, 4, 2027, 2028, 2029, 2030, 2031],
#     "ROI Width": [22, 22, 22, 21, 22, 24, 25, 24, 24, 24],
#     "ROI Height": [2, 2, 2, 2, 2, 4, 3, 4, 4, 3],
#     "Area": [189, 189, 189, 165, 189, 222, 221, 222, 222, 216],
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# Plot distributions for each column
columns_to_plot = ["Frame Number", "ROI Width", "ROI Height", "Area"]
for column in columns_to_plot:
    plt.figure(figsize=(8, 4))
    
    # Histogram and Density Plot
    sns.histplot(meta_df[column], kde=True, bins=30, color="skyblue", label=f"Distribution of {column}")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# %%
