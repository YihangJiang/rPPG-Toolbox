# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Directory containing the CSV and video files
data_dir = '/hpc/group/dunnlab/rppg_data/data/DATASET_1_IN/s2'

# Get list of CSV and video files
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
vid_files = [f for f in os.listdir(data_dir) if f.endswith('.avi')]

# Summarize CSV files
print("CSV Files Summary:")
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    try:
        bvp_df = pd.read_csv(file_path)
        print(f"\nFile: {csv_file}")
        print(bvp_df.describe())
        print(bvp_df.info())
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

# Summarize Video files
print("\nVideo Files Summary:")
for vid_file in vid_files:
    file_path = os.path.join(data_dir, vid_file)
    try:
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"\nFile: {vid_file}")
        print(f"Total Frames: {frame_count}")
        print(f"Frame Dimensions: {frame_width}x{frame_height}")
        print(f"Frame Rate: {frame_rate} FPS")
    except Exception as e:
        print(f"Error reading {vid_file}: {e}")

# %%
