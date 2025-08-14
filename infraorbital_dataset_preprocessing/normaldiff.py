# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_frame_diff(video_path):
    """Calculate the difference between consecutive frames in a video."""
    
    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get video properties
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Total frames: {n_frames}, Frame dimensions: {width}x{height}")

    # Initialize a list to store frames
    frames = []
    r = 0
    
    # Read the frames from the video
    while True:
        r = r +1
        ret, frame = cap.read()
        if r > 11:
            break
        if not ret:
            break
        frames.append(frame)

    cap.release()  # Release the video capture object

    # Convert frames to numpy array
    frames = np.array(frames)
    
    # Initialize a list to store the differences
    diff_frames = []

    # Calculate the difference between consecutive frames
    for i in range(1, len(frames)):
        print(i)
        # Convert frames to float for the diff calculation
        frame1 = frames[i - 1].astype(np.float32)
        frame2 = frames[i].astype(np.float32)

        # Compute the difference
        diff = (frame2 - frame1) / (frame2 + frame1 + 1e-7)  # Normalized difference

        # Append to the list of differences
        diff_frames.append(diff)

    # Convert the difference frames to numpy array
    diff_frames = np.array(diff_frames)
    
    # Visualize the first 10 frames of the differences (optional)
    visualize_frames(diff_frames[:10])

    return diff_frames

def visualize_frames(frames):
    """Visualize the first few frames."""
    num_frames = len(frames)
    
    # Create a figure with subplots to display the frames
    fig, axes = plt.subplots(1, num_frames, figsize=(20, 5))

    for i in range(num_frames):
        axes[i].imshow(frames[i].astype(np.uint8))  # Convert to uint8 for proper display
        axes[i].axis('off')  # Turn off axis

    plt.tight_layout()
    plt.show()

# Example usage: Replace 'your_video_path.mp4' with the actual path to your video
video_path = '/work/yj167/DATASET_1/s4/vid_s4_T3.avi'  # Path to the video file
diff_frames = calculate_frame_diff(video_path)


# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_frame_diff(video_path, frame_interval=100):
    """Calculate the difference between frames every 'frame_interval' frames in a video."""
    
    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get video properties
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Total frames: {n_frames}, Frame dimensions: {width}x{height}")

    # Initialize a list to store frames
    frames = []
    r = 0
    # Read the frames from the video
    while True:
        r = r + 1
        print(r)
        ret, frame = cap.read()
        if not ret:
            break
        if r > 2000:
            break
        frames.append(frame)

    cap.release()  # Release the video capture object

    # Convert frames to numpy array
    frames = np.array(frames)
    
    # Initialize a list to store the differences
    diff_frames = []

    # Calculate the difference every 'frame_interval' frames
    for i in range(frame_interval, len(frames), frame_interval):
        # Convert frames to float for the diff calculation
        frame1 = frames[i - frame_interval].astype(np.float32)
        frame2 = frames[i].astype(np.float32)

        # Compute the difference
        diff = (frame2 - frame1) / (frame2 + frame1 + 1e-7)  # Normalized difference

        # Append to the list of differences
        diff_frames.append(diff)

    # Convert the difference frames to numpy array
    diff_frames = np.array(diff_frames)
    
    # Visualize the first few frames of the differences (optional)
    visualize_frames(diff_frames[:10])

    return diff_frames

def visualize_frames(frames):
    """Visualize the first few frames."""
    num_frames = len(frames)
    
    # Create a figure with subplots to display the frames
    fig, axes = plt.subplots(1, num_frames, figsize=(20, 5))

    for i in range(num_frames):
        axes[i].imshow(frames[i].astype(np.uint8))  # Convert to uint8 for proper display
        axes[i].axis('off')  # Turn off axis

    plt.tight_layout()
    plt.show()

# Example usage
video_path = '/work/yj167/DATASET_1/s4/vid_s4_T3.avi'  # Path to your video file
diff_frames = calculate_frame_diff(video_path, frame_interval=100)


# %%
