import cv2
import dlib
import numpy as np
import os
from imutils import face_utils

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for EAR below which we consider the eye to be blinking
EAR_THRESHOLD = 0.2

# Function to determine if an eye is blinking in an image
def is_blinking(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[42:48]
        rightEye = shape[36:42]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        if leftEAR < EAR_THRESHOLD and rightEAR < EAR_THRESHOLD:
            return True
    return False

# Function to process a directory of images
def process_images(directory):
    non_blinking_images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            if not is_blinking(image):
                non_blinking_images.append(filename)
    return non_blinking_images

# Replace 'path_to_directory' with your actual directory path
path_to_directory = 'photos'
result = process_images(path_to_directory)
print("Images without blinking persons:", result)