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
from collections import deque

# Configuration
MODEL_PATH = 'asl_digit_recognition_model.keras'
IMAGE_SIZE = (128, 128)
FPS_TARGET = 30
SMOOTHING_FRAMES = 7  # Average over last 7 predictions for stability

print("=" * 60)
print("🚀 ASL DIGIT RECOGNITION - REAL-TIME")
print("=" * 60)
print("\n✨ FEATURES:")
print("  • Real-time continuous prediction")
print("  • Temporal smoothing (7 frames) for stability")
print("  • Exact preprocessing match with training")
print("  • Works best when you hold sign steady")
print("=" * 60)

# Load the trained model
print("\n📦 Loading trained model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please train the model first:")
    print("   python train_model.py")
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

# Prediction smoothing buffer
prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)

def get_hand_bbox(hand_landmarks, img_width, img_height, padding=80):
    """Calculate bounding box around detected hand with generous padding"""
    x_coords = [lm.x * img_width for lm in hand_landmarks]
    y_coords = [lm.y * img_height for lm in hand_landmarks]
    
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(img_width, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(img_height, int(max(y_coords)) + padding)
    
    return x_min, y_min, x_max, y_max

def smooth_predictions(new_predictions):
    """
    Add new predictions to buffer and return smoothed result
    Averages predictions over last N frames for stability
    """
    prediction_buffer.append(new_predictions)
    
    if len(prediction_buffer) > 0:
        # Average all predictions in buffer
        avg_predictions = np.mean(prediction_buffer, axis=0)
        predicted_class = np.argmax(avg_predictions)
        confidence = avg_predictions[predicted_class]
        return predicted_class, confidence, avg_predictions
    
    return None, 0.0, None

def preprocess_hand_image(hand_roi):
    """
    Preprocess hand region EXACTLY like training data:
    1. Convert to grayscale
    2. Resize to 128x128
    3. Normalize to [0, 1]
    
    NO CLAHE or other enhancements - must match training!
    """
    # Convert to grayscale
    if len(hand_roi.shape) == 3:
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = hand_roi
    
    # Resize to target size (same as training)
    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    preprocessed = np.expand_dims(normalized, axis=[0, -1])
    
    return preprocessed

# Initialize webcam
print("\n📹 Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit(1)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

print("✅ Camera initialized")
print("\n✅ System ready!")
print("\n" + "=" * 60)
print("INSTRUCTIONS:")
print("  • Show your hand with a digit sign (0-9)")
print("  • Hold steady for ~0.5 sec for accurate reading")
print("  • Predictions are smoothed over 7 frames for stability")
print("  • Press 'R' to reset prediction buffer")
print("  • Press ESC to exit")
print("=" * 60 + "\n")

frame_count = 0

while True:
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
        bbox = get_hand_bbox(hand_landmarks, w, h, padding=80)
        x_min, y_min, x_max, y_max = bbox
        
        # Extract hand region
        hand_roi = frame[y_min:y_max, x_min:x_max]
        
        if hand_roi.size > 0 and hand_roi.shape[0] > 10 and hand_roi.shape[1] > 10:
            # Preprocess the hand region
            preprocessed = preprocess_hand_image(hand_roi)
            
            # Get raw prediction from model
            raw_predictions = model.predict(preprocessed, verbose=0)[0]
            
            # Apply temporal smoothing (average over last N frames)
            predicted_class, confidence, smoothed_predictions = smooth_predictions(raw_predictions)
            
            if predicted_class is not None:
                # Determine color and status based on confidence
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green - high confidence
                    status = "HIGH"
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow - good confidence
                    status = "GOOD"
                elif confidence > 0.3:
                    color = (0, 165, 255)  # Orange - medium confidence
                    status = "MEDIUM"
                else:
                    color = (0, 0, 255)  # Red - low confidence
                    status = "LOW"
                
                # ALWAYS draw bounding box
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), color, 3)
                
                # ALWAYS draw prediction box
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (x_min, y_min - 110), (x_min + 450, y_min - 10), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)
                
                # ALWAYS draw prediction text
                cv2.putText(display_frame, f"DIGIT: {predicted_class}", 
                           (x_min + 10, y_min - 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(display_frame, f"Confidence: {confidence:.1%} ({status})", 
                           (x_min + 10, y_min - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show buffer size indicator
                buffer_text = f"Smoothing: {len(prediction_buffer)}/{SMOOTHING_FRAMES}"
                cv2.putText(display_frame, buffer_text, (x_min + 10, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # ALWAYS draw top 3 predictions
                top_3_indices = np.argsort(smoothed_predictions)[-3:][::-1]
                y_offset = y_max + 30
                
                for i, idx in enumerate(top_3_indices):
                    pred_text = f"{idx}: {smoothed_predictions[idx]:.1%}"
                    cv2.putText(display_frame, pred_text, (x_min, y_offset + i*30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # No hand detected - clear buffer
        if len(prediction_buffer) > 0:
            prediction_buffer.clear()
        cv2.putText(display_frame, "Show hand with digit sign (0-9)", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Draw instructions at bottom
    height, width = display_frame.shape[:2]
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, height - 50), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
    
    cv2.putText(display_frame, "R: Reset | ESC: Exit", 
               (width//2 - 120, height - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display frame
    cv2.imshow("ASL Digit Recognition - Real-Time", display_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        print("\n👋 Exiting...")
        break
    elif key == ord('r') or key == ord('R'):  # Reset buffer
        prediction_buffer.clear()
        print("🔄 Prediction buffer reset")
    
    frame_count += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n✅ Session ended")
print("=" * 60)
