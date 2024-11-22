import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def plot_landmark(img_base, facial_area_name, facial_area_obj, landmarks):
    """_summary_

    Args:
        img_base (_type_): _description_
        facial_area_name (_type_): _description_
        facial_areas_connect (_type_): _description_
    """

    print(facial_area_name, ":")

    img = img_base.copy()
    for source_idx, target_idx in facial_area_obj:
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]

        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
        cv2.circle(img, relative_source, 10, (255, 0, 0), -1)
        cv2.circle(img, relative_target, 10, (255, 0, 0), -1)
        # cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 10)

    
    fig = plt.figure(figsize = (15, 15))
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

def plot_semi(image, pt_1, pt_2):
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

    # Plot the two original points for reference
    plt.plot([x1, x2], [y1, y2], 'bo')  # Plot the two points in blue for clarity

    # Mask the semi-circle area (all pixels within the semi-circle)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if (x - mid_x)**2 + (y - mid_y)**2 <= (radius)**2:  # Inside the circle
                if y < mid_y:  # Only the lower semi-circle
                    mask[y, x] = 0  # Mark as within semi-circle

    # Extract pixel values within the semi-circle
    extracted_pixels = image[mask == 255]

    # Display the masked semi-circle area for visualization
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    plt.imshow(masked_image)
    plt.title("Semi-circle Mask on Image")
    plt.grid()
    plt.show()

    plt.imshow(extracted_pixels)

def locate_eye_corner(results_face_mesh, eye_seq, img):
  x_list = [results_face_mesh.multi_face_landmarks[0].landmark[i].x for i in eye_seq]
  y_list = [results_face_mesh.multi_face_landmarks[0].landmark[i].y for i in eye_seq]

  min_id, max_id = x_list.index(min(x_list)), x_list.index(max(x_list))
  pt_min = (int(x_list[min_id] * img.shape[1]), int(y_list[min_id] * img.shape[0]))
  pt_max = (int(x_list[max_id] * img.shape[1]), int(y_list[max_id] * img.shape[0])) 
  return pt_min, pt_max

def get_seq_num_facial_areas(facial_areas, area_name):
  
