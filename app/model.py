import numpy as np
import tensorflow as tf
from image_processing import preprocess_image
from PIL import Image

# Model und Label laden
model = tf.keras.models.load_model('model/202411111243_asl_sign_language_model.keras')
label_classes = np.load('model/202411111243_label_classes.npy')

def predict(image):
    annotated_image, landmarks = preprocess_image(image)
    if landmarks is None:
        return annotated_image, {"Letter": "No Hand Found", "Confidence": 0.0}

    predictions = model.predict(landmarks)
    confidence = float(np.max(predictions) * 100)
    label = str(label_classes[np.argmax(predictions)])

    return annotated_image, {"Letter": label, "Confidence": confidence}
