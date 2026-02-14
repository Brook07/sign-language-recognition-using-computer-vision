import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd

# Create hand landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

DATASET_PATH = r"dataset\American_Sign_Language_Digits_Dataset"
data = []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue

    input_images_path = os.path.join(label_path, f"Input Images - Sign {label}")
    if not os.path.isdir(input_images_path):
        continue

    print(f"Processing sign {label}...")
    for img_name in os.listdir(input_images_path):
        img_path = os.path.join(input_images_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks.append(label)  # add label
            data.append(landmarks)

detector.close()

# Create column names
columns = []
for i in range(21):
    columns += [f"x{i}", f"y{i}", f"z{i}"]
columns.append("label")

df = pd.DataFrame(data, columns=columns)
df.to_csv("asl_digits_landmarks.csv", index=False)

print("✅ Landmark dataset saved as asl_digits_landmarks.csv")
