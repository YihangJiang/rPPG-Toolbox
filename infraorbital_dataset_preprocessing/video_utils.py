import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import re
import os


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

def plot_rois(results, image):
  # Check if a face was detected
  if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
          # Get image dimensions
          h, w, _ = image.shape
          overlay = image.copy()

          # Iterate through each facial region and draw it
          for region, keypoints_indices in zip(region_names, keypoints):
              points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in keypoints_indices]

              # Draw the polygon for each region
              cv2.polylines(overlay, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

          # Display the image with the highlighted regions
          plt.figure(figsize=(8, 8))
          plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
          plt.axis("off")
          plt.title("Highlighted Facial Regions")
          plt.show()
  else:
      print("No face detected. Please try a different image.")

def mod_mask(mask, pt_1, pt_2):
    # Define the two points (peaks) for the semi-circle
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

    m = (y2 - y1) / (x2 - x1)  # slope
    b = y1 - m * x1  # y-intercept
    # Mask the semi-circle area (all pixels within the semi-circle)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if (x - mid_x)**2 + (y - mid_y)**2 <= (radius)**2:  # Inside the circle
                if y < m*x+b:  # Only the lower semi-circle
                    mask[y, x] = 0  # Mark as within semi-circle

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
  directory_path = os.path.dirname(output_video_path)
  if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory created: {directory_path}")
  else:
    print(f"Directory already exists: {directory_path}")

  return output_video_path

def segment_one_video(video_path, output_video_path, area_names):
    subject = re.search(r'subject(\d+)', video_path).group(1)
    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    with tqdm(total=frame_count, desc=f"Processing {subject}", unit="frame") as pbar:
      while(cap.isOpened()):
          success, image = cap.read()
          if not success:
            print("End of frame")
            break

          results = face_detection(image)
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
  


  
