# %%
%reload_ext autoreload
%autoreload 2

import cv2
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import mediapipe as mp
from video_utils import *
import pandas as pd
area_names = ['Right_eye', 'Left_eye']

# %% Extract the names of the path
# Path to the main folder containing subject subfolders
main_folder = '/hpc/group/dunnlab/rppg_data/data/DATASET_2'
subject_pattern = re.compile(r'^subject\d+$')
subject_path_list = []
video_path_list = []

# Loop through each subject folder, save the directory path of each raw video
for subject_folder in os.listdir(main_folder):
    subject_path = os.path.join(main_folder, subject_folder)
    if os.path.isdir(subject_path) and subject_pattern.match(subject_folder):  # Check if it's a directory
        subject_path_list.append(subject_path)
        for file in os.listdir(subject_path):
            if file.endswith('.avi'):  # Check if the file is a video
                video_path = os.path.join(subject_path, file)
                video_path_list.append(video_path)

            if file.endswith(".txt"):
                ground_truth_path = os.path.join(subject_path, file)
                output_dir_path = re.sub(r'(DATASET_\d)', r'\1_IN', subject_path)
                shutil.copy(ground_truth_path, output_dir_path)


# %% Use directory path to save the videos that only contain the infraorbital region
for i in range(1):
    i = 3 
    output_path = check_output_path(video_path_list[i])
    segmented_frames = segment_one_video(video_path_list[i], output_path, area_names)


# %%
