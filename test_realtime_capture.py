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
from collections import Counter

# Configuration
MODEL_PATH = 'asl_digit_recognition_model.keras'
IMAGE_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.60
FPS_TARGET = 30

print("=" * 60)
print("🚀 ASL DIGIT RECOGNITION - SPACE CAPTURE MODE")
print("=" * 60)
print("\n✨ FEATURES:")
print("  • Press SPACE to start capturing frames")
print("  • Press SPACE again to stop and process")
print("  • Result based on most frequent prediction")
print("  • Process multiple frames as static images")
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

def get_hand_bbox(hand_landmarks, img_width, img_height, padding=50):
    """
    Calculate bounding box around detected hand with extra padding
    """
    x_coords = [lm.x * img_width for lm in hand_landmarks]
    y_coords = [lm.y * img_height for lm in hand_landmarks]
    
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(img_width, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(img_height, int(max(y_coords)) + padding)
    
    return x_min, y_min, x_max, y_max

def preprocess_hand_image_like_training(hand_roi):
    """
    Preprocess hand region EXACTLY like training data preprocessing
    """
    # Convert to grayscale
    if len(hand_roi.shape) == 3:
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = hand_roi
    
    # Resize to target size
    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    preprocessed = np.expand_dims(normalized, axis=[0, -1])
    
    return preprocessed, resized

def process_captured_frames(predictions_list):
    """
    Process all captured predictions and return the most frequent one
    """
    if not predictions_list:
        return None, 0.0, {}
    
    # Count frequency of each predicted digit
    digit_counter = Counter(predictions_list)
    
    # Get the most common digit
    most_common_digit, frequency = digit_counter.most_common(1)[0]
    
    # Calculate confidence as percentage of frames
    confidence = frequency / len(predictions_list)
    
    # Create frequency distribution
    frequency_dist = {digit: count/len(predictions_list) for digit, count in digit_counter.items()}
    
    return most_common_digit, confidence, frequency_dist

def draw_capture_ui(frame, capturing, predictions_count, last_result=None):
    """
    Draw capture mode UI
    """
    height, width = frame.shape[:2]
    
    # Draw semi-transparent top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    if capturing:
        # Capturing mode
        status_text = "🔴 CAPTURING..."
        color = (0, 0, 255)
        instruction = f"Frames captured: {predictions_count} | Press SPACE to STOP"
        
        # Flashing border
        if predictions_count % 10 < 5:
            cv2.rectangle(frame, (10, 10), (width-10, height-10), (0, 0, 255), 5)
    else:
        # Ready mode
        status_text = "⚪ READY"
        color = (0, 255, 0)
        instruction = "Press SPACE to START capturing"
    
    cv2.putText(frame, status_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(frame, instruction, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw bottom panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw result if available
    if last_result:
        digit, conf, freq_dist = last_result
        result_text = f"RESULT: Digit {digit} ({conf:.0%} confidence)"
        cv2.putText(frame, result_text, (20, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Show frequency distribution
        freq_text = " | ".join([f"{d}:{p:.0%}" for d, p in sorted(freq_dist.items())])
        cv2.putText(frame, freq_text, (20, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "ESC: Exit | R: Reset", (20, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_hand_bbox(frame, bbox, prediction=None, confidence=None):
    """
    Draw bounding box around detected hand
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Draw bounding box
    color = (0, 255, 0)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    
    if prediction is not None and confidence is not None:
        # Draw current prediction
        text = f"{prediction}: {confidence:.1%}"
        cv2.putText(frame, text, (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
print("  1. Show your hand with a digit sign (0-9)")
print("  2. Press SPACE to START capturing frames")
print("  3. Hold your hand steady while capturing")
print("  4. Press SPACE to STOP and see the result")
print("  5. Result is based on most frequent prediction")
print("  6. Press R to reset and try again")
print("  7. Press ESC to exit")
print("=" * 60 + "\n")

# State variables
capturing = False
captured_predictions = []
captured_confidences = []
last_result = None
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
    
    current_prediction = None
    current_confidence = None
    current_bbox = None
    
    if detection_result.hand_landmarks:
        # Get first hand
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Get bounding box
        h, w = frame.shape[:2]
        bbox = get_hand_bbox(hand_landmarks, w, h, padding=50)
        x_min, y_min, x_max, y_max = bbox
        current_bbox = bbox
        
        # Extract hand region
        hand_roi = frame[y_min:y_max, x_min:x_max]
        
        if hand_roi.size > 0 and hand_roi.shape[0] > 10 and hand_roi.shape[1] > 10:
            # Preprocess like training data
            preprocessed, processed_gray = preprocess_hand_image_like_training(hand_roi)
            
            # Get prediction - process as static image
            predictions = model.predict(preprocessed, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            
            current_prediction = predicted_class
            current_confidence = confidence
            
            # If capturing, store the prediction
            if capturing and confidence > CONFIDENCE_THRESHOLD:
                captured_predictions.append(predicted_class)
                captured_confidences.append(confidence)
            
            # Draw hand bounding box with current prediction
            draw_hand_bbox(display_frame, bbox, predicted_class, confidence)
            
            # Show preprocessed image in corner
            debug_size = (150, 150)
            debug_img = cv2.resize(processed_gray, debug_size)
            debug_img_bgr = cv2.cvtColor((debug_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            y_offset = 130
            display_frame[y_offset:y_offset+debug_size[0], 10:10+debug_size[1]] = debug_img_bgr
            cv2.putText(display_frame, "Preprocessed", (15, y_offset + debug_size[0] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        # No hand detected
        if capturing:
            cv2.putText(display_frame, "⚠️ No hand detected!", 
                       (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Draw capture UI
    draw_capture_ui(display_frame, capturing, len(captured_predictions), last_result)
    
    # Display frame
    cv2.imshow("ASL Digit Recognition - Capture Mode", display_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        print("\n👋 Exiting...")
        break
    
    elif key == 32:  # SPACE
        if not capturing:
            # Start capturing
            capturing = True
            captured_predictions = []
            captured_confidences = []
            last_result = None
            print("\n🔴 Started capturing frames...")
        else:
            # Stop capturing and process result
            capturing = False
            print(f"\n⏹️  Stopped capturing. Captured {len(captured_predictions)} frames")
            
            if captured_predictions:
                # Process captured frames
                final_digit, final_confidence, freq_dist = process_captured_frames(captured_predictions)
                last_result = (final_digit, final_confidence, freq_dist)
                
                # Display result
                print("\n" + "=" * 60)
                print("📊 CAPTURE RESULTS:")
                print("=" * 60)
                print(f"Total frames captured: {len(captured_predictions)}")
                print(f"\n🎯 FINAL RESULT: Digit {final_digit}")
                print(f"Confidence: {final_confidence:.1%} ({int(final_confidence * len(captured_predictions))} out of {len(captured_predictions)} frames)")
                print(f"Average prediction confidence: {np.mean(captured_confidences):.1%}")
                print(f"\nFrequency distribution:")
                for digit, percentage in sorted(freq_dist.items()):
                    bar = "█" * int(percentage * 30)
                    print(f"  Digit {digit}: {percentage:>6.1%} {bar}")
                print("=" * 60 + "\n")
            else:
                print("⚠️  No valid predictions captured!")
    
    elif key == ord('r') or key == ord('R'):
        # Reset
        capturing = False
        captured_predictions = []
        captured_confidences = []
        last_result = None
        print("\n🔄 Reset complete - Ready to capture")
    
    frame_count += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n✅ Session ended")
print("=" * 60)
