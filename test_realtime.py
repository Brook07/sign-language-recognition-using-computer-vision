import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow import keras

# Configuration
MODEL_PATH = 'asl_digit_recognition_model.keras'
IMAGE_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to show prediction

print("=" * 60)
print("🚀 ASL DIGIT RECOGNITION - REAL-TIME TESTING")
print("=" * 60)

# Load the trained model
print("\n📦 Loading trained model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please make sure you have trained the model first (run train_cnn.py)")
    exit(1)

# Initialize MediaPipe Hand Detector
print("\n🖐️  Initializing hand detector...")
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

def get_hand_bbox(hand_landmarks, img_width, img_height, padding=40):
    """
    Calculate bounding box around detected hand with padding
    """
    x_coords = [lm.x * img_width for lm in hand_landmarks]
    y_coords = [lm.y * img_height for lm in hand_landmarks]
    
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(img_width, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(img_height, int(max(y_coords)) + padding)
    
    return x_min, y_min, x_max, y_max

def preprocess_hand_image(hand_roi):
    """
    Preprocess hand region for model prediction
    """
    # Convert to grayscale
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    
    # Resize to model input size
    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    preprocessed = np.expand_dims(normalized, axis=[0, -1])
    
    return preprocessed

def draw_prediction_ui(frame, prediction, confidence, bbox):
    """
    Draw prediction information on frame
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Draw bounding box
    color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    
    # Prepare text
    if confidence > CONFIDENCE_THRESHOLD:
        text = f"Sign: {prediction}"
        conf_text = f"Conf: {confidence:.2%}"
    else:
        text = "Low Confidence"
        conf_text = f"{confidence:.2%}"
    
    # Draw background rectangles for text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    # Main prediction background
    cv2.rectangle(frame, 
                  (x_min, y_min - 50), 
                  (x_min + text_size[0] + 10, y_min - 10),
                  color, -1)
    
    # Confidence background
    cv2.rectangle(frame, 
                  (x_min, y_min - 80), 
                  (x_min + conf_size[0] + 10, y_min - 55),
                  (50, 50, 50), -1)
    
    # Draw text
    cv2.putText(frame, text, (x_min + 5, y_min - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, conf_text, (x_min + 5, y_min - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_info_panel(frame):
    """
    Draw information panel on frame
    """
    height, width = frame.shape[:2]
    
    # Draw semi-transparent panel at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 60), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Instructions
    instructions = [
        "ESC: Exit",
        "SPACE: Pause",
        "C: Clear",
        "Show hand signs 0-9"
    ]
    
    x_offset = 10
    for instruction in instructions:
        cv2.putText(frame, instruction, (x_offset, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_offset += 200

# Initialize webcam
print("\n📹 Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit(1)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("\n✅ System ready!")
print("\n" + "=" * 60)
print("INSTRUCTIONS:")
print("  • Show your hand with a digit sign (0-9)")
print("  • Press SPACE to pause/resume")
print("  • Press ESC to exit")
print("=" * 60 + "\n")

paused = False
frame_count = 0
prediction_text = ""
confidence_value = 0.0

while True:
    if not paused:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read from webcam")
            break
        
        frame = cv2.flip(frame, 1)  # Mirror the frame
        display_frame = frame.copy()
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detect hand
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            # Get first hand
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Get bounding box
            h, w = frame.shape[:2]
            bbox = get_hand_bbox(hand_landmarks, w, h, padding=40)
            x_min, y_min, x_max, y_max = bbox
            
            # Extract hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]
            
            if hand_roi.size > 0:
                # Preprocess and predict
                preprocessed = preprocess_hand_image(hand_roi)
                predictions = model.predict(preprocessed, verbose=0)[0]
                
                # Get prediction and confidence
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class]
                
                prediction_text = str(predicted_class)
                confidence_value = confidence
                
                # Draw prediction UI
                draw_prediction_ui(display_frame, prediction_text, confidence, bbox)
        else:
            # No hand detected
            cv2.putText(display_frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw info panel
        draw_info_panel(display_frame)
        
        # Show FPS
        frame_count += 1
        if frame_count % 30 == 0:
            frame_count = 0
    
    else:
        # Paused mode
        cv2.putText(display_frame, "PAUSED - Press SPACE to resume", 
                   (display_frame.shape[1]//2 - 200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Display frame
    cv2.imshow("ASL Digit Recognition - Real-Time", display_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        print("\n👋 Exiting...")
        break
    elif key == 32:  # SPACE
        paused = not paused
        print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
    elif key == ord('c') or key == ord('C'):
        prediction_text = ""
        confidence_value = 0.0
        print("🗑️  Cleared predictions")

# Cleanup
cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n✅ Session ended")
print("=" * 60)
