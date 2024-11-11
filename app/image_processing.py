import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from landmarks import normalize_landmarks, adjust_brightness_contrast

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)

def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_adjusted = adjust_brightness_contrast(image, brightness=20)
    results = hands.process(cv2.cvtColor(image_adjusted, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image_adjusted, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        normalized_landmarks = normalize_landmarks(landmarks)
        landmarks_array = np.array(normalized_landmarks).reshape(1, 21, 3, 1)
        return image_adjusted, landmarks_array
    
    return image, None
