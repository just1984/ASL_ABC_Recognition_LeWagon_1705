import cv2
import mediapipe as mp
import numpy as np
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
from datetime import datetime

# TRAIN_ALPHABET_DIR = "/home/just161/code/SIGN/new/sign_language_interpreter/training_data/Train_ABC_NUM"
TRAIN_ALPHABET_DIR = "/home/just161/code/SIGN/new/sign_language_interpreter/training_data/TEST"

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001
PATIENCE = 5

datagen = ImageDataGenerator(
    brightness_range=[0.8, 1.2],
    zoom_range=0.6,
    rotation_range=50,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

# print("\n\n##### ##### ##### ##### GPU available:", tf.config.list_physical_devices('GPU'))
# with tf.device('/GPU:0'):
#     start = time.time()
#     a = tf.random.normal([10000, 10000])
#     b = tf.random.normal([10000, 10000])
#     c = tf.matmul(a, b)
#     print("\n\n##### ##### ##### ##### Berechnungszeit auf der GPU:", time.time() - start)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

data_dir = TRAIN_ALPHABET_DIR
landmark_data = []
labels = []

save_interval = 25

output_dir = "output"
output_dir_augmented = os.path.join(output_dir, "augmented_samples")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_augmented, exist_ok=True)

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

data_count = {}
for letter in tqdm(os.listdir(data_dir), desc="Processing Letters"):
    letter_dir = os.path.join(data_dir, letter)
    count = 0
    for i, img_path in enumerate(os.listdir(letter_dir)):
        file_extension = img_path.lower().rsplit('.', 1)[-1]
        
        if file_extension not in {'jpg', 'jpeg', 'png'}:
            continue
            
        img = cv2.imread(os.path.join(letter_dir, img_path))
        if img is None:
            print(f"Skipped: {img_path}") 
            continue
            
        img_resized = cv2.resize(img, (224, 224))
        img_augmented = datagen.random_transform(img_resized)

        img_augmented = img_augmented.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_augmented, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = [coord for lm in results.multi_hand_landmarks[0].landmark for coord in (lm.x, lm.y, lm.z)]
            landmarks = normalize_landmarks(landmarks)
            landmark_data.append(landmarks)
            labels.append(letter)
            count += 1

            if count % save_interval == 0:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img_augmented, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                save_path = os.path.join(output_dir_augmented, f"{letter}_{i}.png")
                cv2.imwrite(save_path, cv2.cvtColor(img_augmented, cv2.COLOR_RGB2BGR))
                
    data_count[letter] = count
    print(f"{letter}: {count} images")

print("\nFound Data for each Letter / Number:")
for key, value in data_count.items():
    print(f"{key}: {value} Images")

timestamp = datetime.now().strftime("%Y%m%d%H%M")

landmark_data = np.array(landmark_data)
labels = np.array(labels)
np.save(os.path.join(output_dir, f"{timestamp}_landmark_data.npy"), landmark_data)
np.save(os.path.join(output_dir, f"{timestamp}_labels.npy"), labels)

landmark_data = landmark_data / np.max(landmark_data)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

X_train, X_test, y_train, y_test = train_test_split(landmark_data, labels_categorical, test_size=0.2, random_state=42)

if len(X_train.shape) == 2:
    height, width = 21, 3
    X_train = X_train.reshape(-1, height, width, 1)
    X_test = X_test.reshape(-1, height, width, 1)

input_shape = (X_train.shape[1], X_train.shape[2], 1)

inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.4)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.4)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.6)(x)
outputs = Dense(len(label_encoder.classes_), activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
              loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(patience=PATIENCE, restore_best_weights=True)

model_filename = os.path.join(output_dir, f"{timestamp}_asl_sign_language_model.keras")
label_classes_filename = os.path.join(output_dir, f"{timestamp}_label_classes.npy")

print("\n\n\nTraining - More Passion More Energy:")
model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[es]
)

save_model(model, model_filename)
print(f"Model saved as: {model_filename}")

np.save(label_classes_filename, label_encoder.classes_)
print(f"Saved Label Classes as: {label_classes_filename}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

present_classes = sorted(set(y_true_classes) | set(y_pred_classes))
target_names = [label_encoder.classes_[i] for i in present_classes]

report = classification_report(y_true_classes, y_pred_classes, target_names=target_names)
print("\n\n\nClassification Report:")
print(report)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for ASL Model")
plt.show()