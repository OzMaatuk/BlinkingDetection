import cv2
import dlib
import os
import numpy as np
from imutils import face_utils

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # You need to download this pre-trained model

def is_blinking(eye_points, vertical_ratio_threshold=0.25):
    # Compute the eye aspect ratio to determine if the eye is closed
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear < vertical_ratio_threshold

def detect_blinking_in_images(image_dir):
    images_with_blinking = []

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            leftEye = shape[42:48]
            rightEye = shape[36:42]
            
            left_blinking = is_blinking(leftEye)
            right_blinking = is_blinking(rightEye)
            
            if (left_blinking or right_blinking):
                images_with_blinking.append(image_path)
                break  # No need to check other faces if one face is not blinking
        
    return images_with_blinking

# Example usage
image_dir = 'path_to_your_directory'
images_without_blinking = detect_blinking_in_images(image_dir)
print('Images without blinking people:')
print(images_without_blinking)