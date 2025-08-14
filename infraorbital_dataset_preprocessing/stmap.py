# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_stmap(video_path, region="whole", resize=(72, 72)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if region == "whole":
            mean_rgb = np.mean(frame, axis=(0, 1))  # R, G, B mean
        else:
            raise NotImplementedError("Only 'whole' frame STMap supported in this example.")

        frames.append(mean_rgb)

    cap.release()

    # Convert to numpy and transpose to get channels as rows
    stmap = np.array(frames).T  # shape: (3, num_frames)

    # Plot STMap
    plt.figure(figsize=(10, 3))
    plt.imshow(stmap, aspect='auto', cmap='jet')
    plt.title("Spatio-Temporal Map (RGB)")
    plt.xlabel("Frame")
    plt.ylabel("Channel (R/G/B)")
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.show()

    return stmap

# Example usage
stmap = generate_stmap("/hpc/group/dunnlab/rppg_data/data/DATASET_2_IN/subject3/vid.avi")

# %%
