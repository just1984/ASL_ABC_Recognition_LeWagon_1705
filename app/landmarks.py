import numpy as np

def normalize_landmarks(landmarks):
    x_coords = landmarks[::3]
    y_coords = landmarks[1::3]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    bbox_width = max_x - min_x
    bbox_height = max_y - min_y

    normalized_landmarks = []
    for i in range(0, len(landmarks), 3):
        normalized_landmarks.append((landmarks[i] - min_x) / bbox_width)
        normalized_landmarks.append((landmarks[i + 1] - min_y) / bbox_height)
        normalized_landmarks.append(landmarks[i + 2])

    return normalized_landmarks

def adjust_brightness_contrast(image, brightness=40, contrast=1.0):
    img = image.astype(np.float32)
    img = img * contrast + brightness
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
