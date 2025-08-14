import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import re
import os
import time
import pandas as pd
from scipy.signal import welch
from scipy.fft import fft



DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

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

keypoints = [
    [10, 109, 108, 151, 337, 338],
    [67, 103, 104, 105, 66, 107, 108, 109],
    [297, 338, 337, 336, 296, 334, 333, 332],
    [151, 108, 107, 55, 8, 285, 336, 337],
    [8, 55, 193, 122, 196, 197, 419, 351, 417, 285],
    [197, 196, 3, 51, 5, 281, 248, 419],
    [4, 45, 134, 220, 237, 44, 1, 274, 457, 440, 363, 275],
    [134, 131, 49, 102, 64, 219, 218, 237, 220],
    [363, 440, 457, 438, 439, 294, 331, 279, 360],
    [5, 51, 45, 4, 275, 281],
    [3, 217, 126, 209, 131, 134],
    [248, 363, 360, 429, 355, 437],
    [188, 114, 217, 236, 196],
    [412, 419, 456, 437, 343],
    [2, 97, 167, 37, 0, 267, 393, 326],
    [97, 165, 185, 40, 39, 37, 167],
    [326, 393, 267, 269, 270, 409, 391],
    [97, 98, 203, 186, 185, 165],
    [326, 391, 409, 410, 423, 327],
    [54, 21, 162, 127, 116, 143, 156, 70, 63, 68],
    [284, 298, 293, 300, 383, 372, 345, 356, 389, 251],
    [126, 100, 118, 117, 116, 123, 147, 187, 205, 203, 129, 209],
    [355, 429, 358, 423, 425, 411, 376, 352, 345, 346, 347, 329],
    [203, 205, 187, 147, 177, 215, 138, 172, 136, 135, 212, 186, 206],
    [423, 426, 410, 432, 364, 365, 397, 367, 435, 401, 376, 411, 425],
    [18, 83, 182, 194, 32, 140, 176, 148, 152, 377, 400, 369, 262, 418, 406, 313],
    [57, 212, 210, 169, 150, 149, 176, 140, 204, 43],
    [287, 273, 424, 369, 400, 378, 379, 394, 430, 432]
]

region_names = [
    "medial forehead", "left lateral forehead", "right lateral forehead", "glabella",
    "upper nasal dorsum", "lower nasal dorsum", "soft triangle", "left ala", "right ala",
    "nasal tip", "left lower nasal sidewall", "right lower nasal sidewall", "left mid nasal sidewall",
    "right mid nasal sidewall", "philtrum", "left upper lip", "right upper lip", "left nasolabial fold",
    "right nasolabial fold", "left temporal", "right temporal", "left malar", "right malar",
    "left lower cheek", "right lower cheek", "chin", "left marionette fold", "right marionette fold"
]

def plot_landmark(img_base, facial_area_name, results, pt_min, pt_max, plot_button):
    """_plot the area of certain parts on the face_

    Args:
        img_base (_type_): _description_
        facial_area_name (_type_): _description_
        facial_areas_connect (_type_): _description_
    """

    landmarks = results.multi_face_landmarks[0]
    facial_area_obj = facial_areas[facial_area_name]
    img = img_base.copy()
    for source_idx, target_idx in facial_area_obj:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
        # cv2.circle(img, relative_source, 10, (255, 0, 0), -1)
        # cv2.circle(img, relative_target, 10, (255, 0, 0), -1)
        # cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 10)

    if plot_button:
      fig = plt.figure(figsize = (15, 15))
      cv2.circle(img, pt_min, 10, (255, 0, 0), -1)
      cv2.circle(img, pt_max, 10, (255, 0, 0), -1)
      plt.axis('off')
      plt.imshow(img[:, :, ::-1])
      plt.show()


def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  plt.imshow(img)
  return img

import cv2
import numpy as np
import pandas as pd

def annotate_video_with_rois(input_video_path, output_video_path, face_mesh, region_name=None, output_size=(320, 320)):
    """
    Warps the specified region from each frame and saves a 320x320 video of the warped patches.
    If the face is not detected in a frame, a black frame is written and its index is saved to a CSV.
    """
    # Open input video
    missed_frames_df = pd.read_csv('./missed_frames.csv', index_col=[0])
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)

    print(f"Output video will be: {output_size[0]}x{output_size[1]} at {fps} FPS")

    # Validate region
    if region_name not in region_names:
        print(f"Region '{region_name}' not found.")
        return
    region_idx = region_names.index(region_name)
    region_landmarks = keypoints[region_idx]

    # Landmark indices (counter-clockwise)
    selected_points = [0, 3, 6, 8]  # [TL, BL, BR, TR]

    # Destination square (counter-clockwise)
    pts_dst = np.float32([
        [0, 0],
        [0, output_size[1] - 1],
        [output_size[0] - 1, output_size[1] - 1],
        [output_size[0] - 1, 0]
    ])

    # Track missed frames
    missed_frames = []
    frame_idx = 0

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        frame_written = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                try:
                    landmark_indices = [region_landmarks[i] for i in selected_points]
                    pts_src = np.float32([
                        [face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
                        for i in landmark_indices
                    ])

                    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
                    warped = cv2.warpPerspective(frame, M, output_size)

                    out_writer.write(warped)
                    frame_written = True
                    break  # Only write first detected face
                except Exception as e:
                    print(f"Frame {frame_idx}: Error warping region: {e}")

        if not frame_written:
            print(f"Frame {frame_idx}: No face detected.")
            missed_frames.append(frame_idx)
            black_frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            out_writer.write(black_frame)

    # Clean up
    cap.release()
    out_writer.release()
    face_mesh.close()

    # Save missed frame indices
    if missed_frames:
        if len(missed_frames_df) == 0:
           missed_frames_df.loc[input_video_path, 'missed_frame'] = missed_frames
        elif pd.Series(input_video_path).isin(missed_frames_df['file_path']).any():
           missed_frames_df.loc[missed_frames_df['file_path']==input_video_path, 'missed_frame'] = missed_frames
        else:
           missed_frames_df.loc[len(missed_frames_df)] = [input_video_path, missed_frames]
    print(missed_frames_df)
    missed_frames_df.to_csv('./missed_frames.csv')
    print(f"Warped region video saved to: {output_video_path}")



# def annotate_video_with_rois(input_video_path, output_video_path, face_mesh, region_names_to_keep, region_names=region_names, keypoints=keypoints):
#     # Open input video
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print("Error opening video file.")
#         return

#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     print(f"Video resolution: {width}x{height} at {fps} FPS")

#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_idx += 1
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_frame)

#         # Start with a black mask and black output frame
#         mask = np.zeros((height, width), dtype=np.uint8)
#         output_frame = np.zeros_like(frame)

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 for region, keypoints_indices in zip(region_names, keypoints):
#                     if region in region_names_to_keep:
#                         points = [
#                             (int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height))
#                             for i in keypoints_indices
#                         ]
#                         cv2.fillPoly(mask, [np.array(points, np.int32)], color=255)

#             # Apply the mask to keep only selected regions
#             output_frame = cv2.bitwise_and(frame, frame, mask=mask)

#         out_writer.write(output_frame)

#     cap.release()
#     out_writer.release()
#     face_mesh.close()
#     print(f"Annotated video saved to: {output_video_path}")


# def annotate_video_with_rois(input_video_path, output_video_path, face_mesh, region_names = region_names, keypoints = keypoints):
#     # Read input video
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print("Error opening video file.")
#         return

#     # Get video properties
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps    = cap.get(cv2.CAP_PROP_FPS)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#     print(width)
#     c = 0
#     while True:
#         ret, frame = cap.read()
#         c = c+1
#         if not ret:
#             break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_frame)
#         overlay = frame.copy()

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 for region, keypoints_indices in zip(region_names, keypoints):
#                     points = [(int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height)) for i in keypoints_indices]
#                     cv2.polylines(overlay, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

#         out_writer.write(overlay)

#     cap.release()
#     out_writer.release()
#     face_mesh.close()
#     print(f"Annotated video saved to: {output_video_path}")


# def plot_rois(results, image, region_names_to_plot=None):
#     """
#     Mask out everything except the specified region(s).
    
#     Args:
#         results: MediaPipe face_mesh results object.
#         image: Input BGR image.
#         region_names_to_plot: List or string of region names to keep. If None, keeps all.
#     """
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             h, w, _ = image.shape

#             # Initialize a black mask
#             mask = np.zeros((h, w), dtype=np.uint8)

#             # Prepare list of regions to plot
#             if region_names_to_plot:
#                 if isinstance(region_names_to_plot, str):
#                     region_names_to_plot = [region_names_to_plot]

#                 regions_to_plot = []
#                 for name in region_names_to_plot:
#                     if name in region_names:
#                         idx = region_names.index(name)
#                         regions_to_plot.append((name, keypoints[idx]))
#                     else:
#                         print(f"Region '{name}' not found. Skipping.")
#             else:
#                 regions_to_plot = list(zip(region_names, keypoints))

#             # Fill the mask with white in the desired region(s)
#             for _, keypoints_indices in regions_to_plot:
#                 points = [
#                     (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
#                     for i in keypoints_indices
#                 ]
#                 cv2.fillPoly(mask, [np.array(points, np.int32)], color=255)

#             # Apply the mask to the image
#             masked_image = cv2.bitwise_and(image, image, mask=mask)

#             # Show result
#             plt.figure(figsize=(8, 8))
#             plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
#             title = "Masked Region(s): " + ", ".join([r for r, _ in regions_to_plot])
#             plt.title(title)
#             plt.axis("off")
#             plt.show()
#     else:
#         print("No face detected. Please try a different image.")

def plot_rois(results, image, region_names_to_plot=None):
    """
    Visualize landmark points in specified region(s), no warping.

    Args:
        results: MediaPipe face_mesh results object.
        image: Input BGR image.
        region_names_to_plot: List or string of region names to show.
    """
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            overlay = image.copy()

            if region_names_to_plot:
                if isinstance(region_names_to_plot, str):
                    region_names_to_plot = [region_names_to_plot]

                regions_to_plot = []
                for name in region_names_to_plot:
                    if name in region_names:
                        idx = region_names.index(name)
                        regions_to_plot.append((name, keypoints[idx]))
                    else:
                        print(f"Region '{name}' not found. Skipping.")
            else:
                regions_to_plot = list(zip(region_names, keypoints))

            for region_name, keypoints_indices in regions_to_plot:
                print(f"\nShowing region: {region_name}")
                for j, i in enumerate(keypoints_indices):
                    lm = face_landmarks.landmark[i]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(
                        overlay, str(j), (x + 2, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA
                    )

            # Show image
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title("Labeled Region Points")
            plt.axis("off")
            plt.show()
    else:
        print("No face detected. Please try a different image.")

# def plot_rois(results, image, region_name, output_size=[72, 72]):
#     """
#     Warp 'right malar' region using perspective transform with selected landmark points.
#     """
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             h, w, _ = image.shape

#             idx = region_names.index(region_name)
#             keypoints_indices = keypoints[idx]

#             # Manually selected point indices from this region (as per your selection)
#             selected_points = [0, 3, 6, 8]  # Indices within the keypoints[idx] list

#             # Convert to full landmark indices
#             try:
#                 landmark_indices = [keypoints_indices[i] for i in selected_points]
#             except IndexError:
#                 print("One of the selected point indices is out of bounds.")
#                 return

#             # Get landmark coordinates
#             pts_src = np.float32([
#                 [face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
#                 for i in landmark_indices
#             ])

#             # Define destination square corners (clockwise)
#             pts_dst = np.float32([
#                 [0, 0],
#                 [0, output_size[1]-1],
#                 [output_size[0]-1, output_size[1]-1],
#                 [output_size[0]-1, 0]
#             ])

#             # Get perspective transform and warp
#             M = cv2.getPerspectiveTransform(pts_src, pts_dst)
#             warped = cv2.warpPerspective(image, M, output_size)

#             # Show the warped region
#             plt.figure(figsize=(3, 3))
#             plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
#             plt.title(f"Warped: {region_name}")
#             plt.axis("off")
#             plt.show()
#     else:
#         print("No face detected. Please try a different image.")




def fill_mask(mask, mid_x, mid_y, radius, m, b):
    # Create a grid of coordinates (X, Y) for the entire mask
    y_indices, x_indices = np.indices(mask.shape)

    # Compute the distance from the center (mid_x, mid_y) for each pixel
    distances = (x_indices - mid_x)**2 + (y_indices - mid_y)**2

    # Create a boolean mask for the pixels inside the circle
    inside_circle = distances <= radius**2

    # Create a boolean mask for the lower semi-circle
    lower_semi_circle = y_indices < m * x_indices + b

    # Combine the two conditions: inside the circle and below the line
    final_mask = np.logical_and(inside_circle, lower_semi_circle)

    # Apply the mask to the original mask array
    mask[final_mask] = 0

    return mask


def mod_mask(mask, pt_1, pt_2):
    # Define the two points (peaks) for the semi-circle
    time1 = time.time()
    x1, y1 = pt_1[0], pt_1[1]  # Example point 1
    x2, y2 = pt_2[0], pt_2[1]  # Example point 2
    radius = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2)

    # Calculate the midpoint and radius of the semi-circle
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Draw a filled circle (use the angle to limit to a semi-circle later)
    cv2.circle(mask, (int(mid_x), int(mid_y)), radius, 255, -1)

    # Calculate the angle between the two points
    angle = np.arctan2(y2 - y1, x2 - x1)

    # Generate theta values for the semi-circle (from π to 2π for the bottom half)
    theta = np.linspace(np.pi, 2 * np.pi, 100)

    # Calculate x and y coordinates of the arc to limit to semi-circle
    x_arc = mid_x + radius * np.cos(angle) * np.cos(theta) - radius * np.sin(angle) * np.sin(theta)
    y_arc = mid_y + radius * np.sin(angle) * np.cos(theta) + radius * np.cos(angle) * np.sin(theta)
    time2 = time.time()


    m = (y2 - y1) / (x2 - x1)  # slope
    b = y1 - m * x1  # y-intercept

    # Mask the semi-circle area (all pixels within the semi-circle)
    fill_mask(mask, mid_x, mid_y, radius, m, b)

    return mask

def plot_semi(image, pt_1_list, pt_2_list, plot_button):
    # Load the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = mod_mask(mask, pt_1_list[0], pt_2_list[0])
    mask = mod_mask(mask, pt_1_list[1], pt_2_list[1])
    # Extract pixel values within the semi-circle
    extracted_pixels = image[mask == 255]

    # Display the masked semi-circle area for visualization
    masked_image = cv2.bitwise_and(image, image, mask=mask)


    if plot_button:
      # Plot the two original points for reference
      # plt.plot([x1, x2], [y1, y2], 'bo')  # Plot the two points in blue for clarity

      plt.imshow(masked_image)
      plt.title("Semi-circle Mask on Image")
      plt.grid()
      plt.show()

      plt.imshow(extracted_pixels)
    return masked_image, extracted_pixels

def locate_eye_corner(results_face_mesh, eye_seq, img):
  x_list = [results_face_mesh.multi_face_landmarks[0].landmark[i].x for i in eye_seq]
  y_list = [results_face_mesh.multi_face_landmarks[0].landmark[i].y for i in eye_seq]

  min_id, max_id = x_list.index(min(x_list)), x_list.index(max(x_list))
  pt_min = (int(x_list[min_id] * img.shape[1]), int(y_list[min_id] * img.shape[0]))
  pt_max = (int(x_list[max_id] * img.shape[1]), int(y_list[max_id] * img.shape[0])) 
  return pt_min, pt_max

def get_seq_num_facial_areas(facial_areas, area_name):
  selected_facial_area = facial_areas[area_name]
  seq_num_list = np.array([])
  for i in selected_facial_area:
      seq_num_list = np.append(seq_num_list, i[0])
      seq_num_list = np.append(seq_num_list, i[1])

  seq_num_list = np.unique(seq_num_list.astype(int))

  return seq_num_list

def face_detection(image):
  results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  return results

def check_output_path(video_path):
  output_video_path = re.sub(r'(DATASET_\d)', r'\1_IN', video_path)
  print(output_video_path)
  directory_path = os.path.dirname(output_video_path)
  exist = 0
  if not os.path.exists(directory_path):
    exist = 0
    os.makedirs(directory_path)
    print(f"Directory created: {directory_path}")
  elif os.path.exists(output_video_path):
    exist = 1
    print(f"Directory already exists but file exists: {directory_path}")

  return output_video_path, exist

def segment_one_video(video_path, output_video_path, area_names):
    subject_num = re.search(r's(\d+)', video_path).group(1)
    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    with tqdm(total=frame_count, desc=f"Processing {subject_num}", unit="frame") as pbar:
      while(cap.isOpened()):
          success, image = cap.read()
          if not success:
            print("End of frame")
            break
          
          results = face_detection(image)
          if results.multi_face_landmarks==None:
            print(f"No landmarks detected in frame {pbar.n}. Skipping this frame.")
            masked_image = np.zeros(image.shape[:2], dtype=np.uint8) 
            processed_frames.append(masked_image)
            out.write(masked_image)
            continue
          else:
            annotated_image = image.copy()
            seq_num_list_1 = get_seq_num_facial_areas(facial_areas, area_names[0])
            seq_num_list_2 = get_seq_num_facial_areas(facial_areas, area_names[1])
            pt_min, pt_max = locate_eye_corner(results, seq_num_list_1, annotated_image)
            pt_min_2, pt_max_2 = locate_eye_corner(results, seq_num_list_2, annotated_image)
            # plot_landmark(annotated_image, area_name, results, pt_min, pt_max, True)
            masked_image, extracted_pixels = plot_semi(annotated_image, [pt_min, pt_min_2], [pt_max, pt_max_2], False)
            processed_frames.append(masked_image)
            out.write(masked_image)
            pbar.update(1)
      

    cap.release()
    out.release()

    return processed_frames
  
def estimate_hr_psd(bvp_signal, fs):
    freqs, psd = welch(bvp_signal, fs=fs, nperseg=min(len(bvp_signal), 1024))

    # Define the heart rate frequency range (in Hz)
    min_hr_hz = 0.5  # ~45 BPM
    max_hr_hz = 5.0   # ~180 BPM
    hr_band = (freqs >= min_hr_hz) & (freqs <= max_hr_hz)

    # Find the frequency with the maximum power in the heart rate band
    peak_freq = freqs[hr_band][np.argmax(psd[hr_band])]

    # Convert to beats per minute
    heart_rate_bpm = peak_freq * 60

    return heart_rate_bpm

def estimate_hr_fft(bvp_signal, fs):
    """Estimates heart rate and returns spectrum for plotting."""
    N = len(bvp_signal)
    # bvp_detrended = detrend(bvp_signal)
    bvp_detrended = bvp_signal

    freqs = np.fft.fftfreq(N, d=1/fs)
    fft_spectrum = np.abs(fft(bvp_detrended))**2

    # Filter to positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_spectrum = fft_spectrum[pos_mask]

    # Limit to HR band
    hr_mask = (freqs >= 0.5) & (freqs <= 5.0)
    freqs_hr = freqs[hr_mask]
    spectrum_hr = fft_spectrum[hr_mask]

    if len(freqs_hr) == 0:
        return None, None, None

    peak_idx = np.argmax(spectrum_hr)
    peak_freq = freqs_hr[peak_idx]
    estimated_hr_bpm = peak_freq * 60

    return estimated_hr_bpm, freqs_hr * 60, spectrum_hr  # HR in BPM

def plot_frame_bvp_counts(csv_files, vid_files):
      # Initialize list to store number of rows in each CSV file
    t1_bvp_list, t2_bvp_list, t3_bvp_list = [], [], []
    bvp_counts = []

    # Count the number of rows in each file
    for file in csv_files:
        print(file)
        try:
            df = pd.read_csv(file, header=None)
            bvp_signal = df.iloc[:, 0].dropna().values

            bvp_counts.append(len(df))
            if 'T1' in file:
                t1_bvp_list.append(bvp_signal)
            elif 'T2' in file:
                t2_bvp_list.append(bvp_signal)
            else:
                t3_bvp_list.append(bvp_signal)
        except Exception as e:
            bvp_counts.append(0)

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(bvp_counts, bins=1, edgecolor='black')
    plt.xlabel("Number of BVP Readings")
    plt.ylabel("Frequency")
    plt.title("Histogram of BVP Reading Counts Across Files")
    plt.xticks(sorted(set(bvp_counts)))  # Show only unique frame counts
    plt.tight_layout()
    plt.show()

    # Initialize list to store frame counts
    frame_counts = []

    # Loop through each file and get the frame count
    for file in vid_files:
        try:
            cap = cv2.VideoCapture(file)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_counts.append(frame_count)
            cap.release()
        except Exception as e:
            frame_counts.append(0)

    # Plot histogram of frame counts
    plt.figure(figsize=(8, 5))
    plt.hist(frame_counts, bins=2, edgecolor='black')
    plt.xlabel("Number of Frames")
    plt.ylabel("Frequency")
    plt.title("Histogram of Frame Counts Across Whole Face Video Files")
    plt.xticks(sorted(set(frame_counts)))  # Show only unique frame counts
    plt.tight_layout()
    plt.show()

    return t1_bvp_list, t2_bvp_list, t3_bvp_list

def get_ubfc_paths(src_root, dst_root):
    list_src, list_dst = [], []
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith(".avi"):
                src_path = os.path.join(root, file)

                # Compute relative path from source root
                rel_path = os.path.relpath(src_path, src_root)

                # Compute corresponding destination path
                dst_path = os.path.join(dst_root, rel_path)
                list_src.append(os.path.join(src_root, rel_path))
                list_dst.append(dst_path)

    return list_src, list_dst


  
