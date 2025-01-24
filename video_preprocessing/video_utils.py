import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp

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

def plot_landmark(img_base, facial_area_name, results, pt_min, pt_max, plot_button):
    """_plot the area of certain parts on the face_

    Args:
        img_base (_type_): _description_
        facial_area_name (_type_): _description_
        facial_areas_connect (_type_): _description_
    """

    print(facial_area_name, ":")
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

def plot_semi(image, pt_1, pt_2, plot_button):
    # Load the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display

    # Define the two points (peaks) for the semi-circle
    x1, y1 = pt_1[0], pt_1[1]  # Example point 1
    x2, y2 = pt_2[0], pt_2[1]  # Example point 2
    radius = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2)

    # Calculate the midpoint and radius of the semi-circle
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Create a mask to store the semi-circle area
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

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

    # Extract pixel values within the semi-circle
    extracted_pixels = image[mask == 255]

    # Display the masked semi-circle area for visualization
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    if plot_button:
      # Plot the two original points for reference
      plt.plot([x1, x2], [y1, y2], 'bo')  # Plot the two points in blue for clarity

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
  
