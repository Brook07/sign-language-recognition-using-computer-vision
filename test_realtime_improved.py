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
CONFIDENCE_THRESHOLD = 0.65  # Slightly higher threshold
FPS_TARGET = 60  # Higher FPS for better capture
PREDICTION_SMOOTHING_FRAMES = 5  # Reduced buffer for faster response
SKIP_FRAMES = 8  # Only predict every 3rd frame for better FPS

print("=" * 60)
print("🚀 ASL DIGIT RECOGNITION - IMPROVED REAL-TIME TESTING")
print("=" * 60)
print("\n✨ NEW FEATURES:")
print("  • 60 FPS capture for better frame quality")
print("  • Exact same preprocessing as training data")
print(f"  • Temporal smoothing (averaging last {PREDICTION_SMOOTHING_FRAMES} predictions)")
print(f"  • Smart frame-skipping (predicts every {SKIP_FRAMES + 1} frames)")
print("  • Frame-by-frame static image processing")
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

# Prediction smoothing buffer
prediction_buffer = deque(maxlen=PREDICTION_SMOOTHING_FRAMES)

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
    This matches the preprocessing from preprocess_dataset.py
    
    NOTE: If accuracy is still low, consider retraining with histogram equalization
    applied to both training and real-time data for better lighting invariance
    """
    # Step 1: Convert to grayscale (if not already)
    if len(hand_roi.shape) == 3:
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = hand_roi
    
    # Step 2: Resize to target size using INTER_AREA (same as training)
    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Step 3: Normalize to [0, 1] as float32
    normalized = resized.astype(np.float32) / 255.0
    
    # Step 4: Add batch and channel dimensions [batch, height, width, channels]
    preprocessed = np.expand_dims(normalized, axis=[0, -1])
    
    return preprocessed, resized

def get_smoothed_prediction(predictions):
    """
    Add current predictions to buffer and return smoothed result
    """
    prediction_buffer.append(predictions)
    
    if len(prediction_buffer) > 0:
        # Average predictions over last N frames
        avg_predictions = np.mean(prediction_buffer, axis=0)
        predicted_class = np.argmax(avg_predictions)
        confidence = avg_predictions[predicted_class]
        return predicted_class, confidence, avg_predictions
    
    return None, 0.0, None

def draw_prediction_ui(frame, prediction, confidence, bbox, all_predictions):
    """
    Draw enhanced prediction information on frame
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Draw bounding box with thicker line
    color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
    
    # Prepare text
    if confidence > CONFIDENCE_THRESHOLD:
        text = f"Digit: {prediction}"
        conf_text = f"Confidence: {confidence:.1%}"
    else:
        text = "Low Confidence"
        conf_text = f"{confidence:.1%}"
    
    # Draw semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_min, y_min - 100), (x_min + 300, y_min - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw text with larger font
    cv2.putText(frame, text, (x_min + 10, y_min - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, conf_text, (x_min + 10, y_min - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw top 3 predictions
    if all_predictions is not None:
        top_3_indices = np.argsort(all_predictions)[-3:][::-1]
        y_offset = y_max + 30
        
        for i, idx in enumerate(top_3_indices):
            pred_text = f"{idx}: {all_predictions[idx]:.1%}"
            cv2.putText(frame, pred_text, (x_min, y_offset + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_info_panel(frame, fps, buffer_size, skip_frames):
    """
    Draw enhanced information panel on frame
    """
    height, width = frame.shape[:2]
    
    # Draw semi-transparent panel at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 80), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Statistics and instructions
    stats = [
        f"FPS: {fps:.1f}",
        f"Buffer: {buffer_size}/{PREDICTION_SMOOTHING_FRAMES}",
        f"Predict: 1/{skip_frames + 1} frames",
        "ESC: Exit | SPACE: Pause | R: Reset"
    ]
    
    x_offset = 10
    for stat in stats:
        cv2.putText(frame, stat, (x_offset, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        x_offset += 230

# Initialize webcam
print("\n📹 Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit(1)

# Set camera properties for high-quality capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)  # Try to set 60 FPS

# Get actual FPS (some cameras don't support 60)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"✅ Camera initialized at {actual_fps} FPS (requested {FPS_TARGET} FPS)")

print("\n✅ System ready!")
print("\n" + "=" * 60)
print("INSTRUCTIONS:")
print("  • Show your hand with a digit sign (0-9)")
print(f"  • Predictions every {SKIP_FRAMES + 1} frames for optimal FPS")
print("  • Hold steady for 0.3-0.5 seconds for best results")
print("  • Press SPACE to pause/resume")
print("  • Press R to reset prediction buffer")
print("  • Press ESC to exit")
print("=" * 60 + "\n")

paused = False
frame_count = 0
prediction_frame_counter = 0
last_time = cv2.getTickCount()
current_fps = 0
last_prediction = None
last_confidence = 0.0
last_bbox = None
last_all_preds = None

while True:
    if not paused:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read from webcam")
            break
        
        # Calculate FPS
        current_time = cv2.getTickCount()
        time_diff = (current_time - last_time) / cv2.getTickFrequency()
        current_fps = 1.0 / time_diff if time_diff > 0 else 0
        last_time = current_time
        
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
            
            # Get bounding box with more padding for better capture
            h, w = frame.shape[:2]
            bbox = get_hand_bbox(hand_landmarks, w, h, padding=50)
            x_min, y_min, x_max, y_max = bbox
            
            # Extract hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]
            
            if hand_roi.size > 0 and hand_roi.shape[0] > 10 and hand_roi.shape[1] > 10:
                # Only predict every SKIP_FRAMES+1 frames to improve FPS
                prediction_frame_counter += 1
                
                if prediction_frame_counter % (SKIP_FRAMES + 1) == 0:
                    # Preprocess EXACTLY like training data
                    preprocessed, processed_gray = preprocess_hand_image_like_training(hand_roi)
                    
                    # Get prediction (process like static image)
                    predictions = model.predict(preprocessed, verbose=0)[0]
                    
                    # Apply temporal smoothing
                    predicted_class, confidence, smoothed_predictions = get_smoothed_prediction(predictions)
                    
                    if predicted_class is not None:
                        # Store prediction for reuse
                        last_prediction = predicted_class
                        last_confidence = confidence
                        last_bbox = bbox
                        last_all_preds = smoothed_predictions
                
                # Draw using last prediction (every frame) or new prediction
                if last_prediction is not None and last_bbox is not None:
                    draw_prediction_ui(display_frame, last_prediction, last_confidence, 
                                     last_bbox, last_all_preds)
                
                # Always show preprocessed image for current frame
                if prediction_frame_counter % (SKIP_FRAMES + 1) == 0:
                    # Optional: Show preprocessed image in corner for debugging
                    debug_size = (150, 150)
                    debug_img = cv2.resize(processed_gray, debug_size)
                    debug_img_bgr = cv2.cvtColor((debug_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    display_frame[10:10+debug_size[0], 10:10+debug_size[1]] = debug_img_bgr
                    cv2.putText(display_frame, "Preprocessed", (15, debug_size[0] + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # No hand detected - clear buffer and predictions
            if len(prediction_buffer) > 0 and frame_count % 10 == 0:
                prediction_buffer.clear()
                last_prediction = None
                last_confidence = 0.0
                last_bbox = None
                last_all_preds = None
            
            cv2.putText(display_frame, "No hand detected - Show digit sign (0-9)", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw info panel
        draw_info_panel(display_frame, current_fps, len(prediction_buffer), SKIP_FRAMES)
        
        frame_count += 1
    
    else:
        # Paused mode
        cv2.putText(display_frame, "PAUSED - Press SPACE to resume", 
                   (display_frame.shape[1]//2 - 250, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    
    # Display frame
    cv2.imshow("ASL Digit Recognition - Improved Real-Time", display_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        print("\n👋 Exiting...")
        break
    elif key == 32:  # SPACE
        paused = not paused
        print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
    elif key == ord('r') or key == ord('R'):
        prediction_buffer.clear()
        last_prediction = None
        last_confidence = 0.0
        last_bbox = None
        last_all_preds = None
        prediction_frame_counter = 0
        print("🔄 Prediction buffer reset")

# Cleanup
cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n✅ Session ended")
print("=" * 60)
