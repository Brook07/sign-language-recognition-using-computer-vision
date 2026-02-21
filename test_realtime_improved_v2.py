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
MODEL_PATH = 'asl_digit_recognition_model_improved.keras'
IMAGE_SIZE = (128, 128)

print("=" * 60)
print("📸 ASL DIGIT RECOGNITION - CAPTURE MODE")
print("=" * 60)
print("\n✨ HOW IT WORKS:")
print("  • Live camera feed displays continuously")
print("  • Press SPACEBAR to capture and analyze your sign")
print("  • Processes as static image with hand detection")
print("  • Shows prediction result on screen")
print("=" * 60)

# Load the trained model
print("\n📦 Loading trained model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please train the improved model first:")
    print("   python train_cnn_improved.py")
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

def get_hand_bbox(hand_landmarks, img_width, img_height, padding=60):
    """
    Calculate bounding box around detected hand with padding
    Increased padding for better hand capture
    """
    x_coords = [lm.x * img_width for lm in hand_landmarks]
    y_coords = [lm.y * img_height for lm in hand_landmarks]
    
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(img_width, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(img_height, int(max(y_coords)) + padding)
    
    return x_min, y_min, x_max, y_max

def preprocess_hand_image_EXACT_MATCH(hand_roi):
    """
    Preprocess hand region EXACTLY like training data:
    1. Convert to grayscale
    2. Apply histogram equalization (NEW - for lighting invariance!)
    3. Resize to target size
    4. Normalize to [0, 1]
    
    This MUST match train_cnn_improved.py preprocessing!
    """
    # Step 1: Convert to grayscale (if not already)
    if len(hand_roi.shape) == 3:
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = hand_roi
    
    # Step 2: Apply histogram equalization (CRITICAL for lighting invariance!)
    equalized = cv2.equalizeHist(gray)
    
    # Step 3: Resize to target size using INTER_AREA (same as training)
    resized = cv2.resize(equalized, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Step 4: Normalize to [0, 1] as float32
    normalized = resized.astype(np.float32) / 255.0
    
    # Step 5: Add batch and channel dimensions [batch, height, width, channels]
    preprocessed = np.expand_dims(normalized, axis=[0, -1])
    
    return preprocessed, equalized, resized

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
    ALWAYS show prediction, with color indicating confidence level
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Draw bounding box with dynamic color based on confidence
    if confidence > HIGH_CONFIDENCE_THRESHOLD:
        color = (0, 255, 0)  # Green for high confidence (>60%)
    elif confidence > 0.30:
        color = (0, 200, 200)  # Yellow for medium confidence (30-60%)
    elif confidence > 0.15:
        color = (0, 165, 255)  # Orange for low-medium confidence (15-30%)
    else:
        color = (0, 0, 255)  # Red for very low confidence (<15%)
    
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
    
    # ALWAYS show the prediction - just indicate confidence level
    text = f"Digit: {prediction}"
    
    # Add confidence indicator to the confidence text
    if confidence > HIGH_CONFIDENCE_THRESHOLD:
        conf_status = "HIGH"
    elif confidence > 0.30:
        conf_status = "MEDIUM"
    elif confidence > 0.15:
        conf_status = "LOW"
    else:
        conf_status = "VERY LOW"
    
    conf_text = f"Confidence: {confidence:.1%} ({conf_status})"
    
    # Draw semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_min, y_min - 100), (x_min + 450, y_min - 10), (0, 0, 0), -1)
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
    Draw information panel on frame
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
cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"✅ Camera initialized at {actual_fps} FPS")

print("\n✅ System ready!")
print("\n" + "=" * 60)
print("INSTRUCTIONS:")
print("  • Show your hand with a digit sign (0-9)")
print("  • Hold steady for 0.3-0.5 seconds for best results")
print("  • Model now handles different lighting conditions better!")
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

# Debug window for preprocessed images
show_debug = True

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
            
            # Get bounding box with more padding
            h, w = frame.shape[:2]
            bbox = get_hand_bbox(hand_landmarks, w, h, padding=60)
            x_min, y_min, x_max, y_max = bbox
            
            # Extract hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]
            
            if hand_roi.size > 0 and hand_roi.shape[0] > 10 and hand_roi.shape[1] > 10:
                # Only predict every SKIP_FRAMES+1 frames
                prediction_frame_counter += 1
                
                if prediction_frame_counter % (SKIP_FRAMES + 1) == 0:
                    # Preprocess with EXACT same method as training
                    preprocessed, equalized_img, resized_img = preprocess_hand_image_EXACT_MATCH(hand_roi)
                    
                    # Get prediction
                    predictions = model.predict(preprocessed, verbose=0)[0]
                    
                    # Apply temporal smoothing
                    predicted_class, confidence, smoothed_predictions = get_smoothed_prediction(predictions)
                    
                    if predicted_class is not None:
                        # Store prediction
                        last_prediction = predicted_class
                        last_confidence = confidence
                        last_bbox = bbox
                        last_all_preds = smoothed_predictions
                    
                    # Show debug preprocessing windows
                    if show_debug:
                        # Show equalized image (new step!)
                        debug_size = (150, 150)
                        
                        # Equalized image
                        debug_eq = cv2.resize(equalized_img, debug_size)
                        debug_eq_bgr = cv2.cvtColor(debug_eq, cv2.COLOR_GRAY2BGR)
                        display_frame[10:10+debug_size[0], 10:10+debug_size[1]] = debug_eq_bgr
                        cv2.putText(display_frame, "Equalized", (15, debug_size[0] + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Resized/preprocessed image
                        debug_res = cv2.resize(resized_img, debug_size)
                        debug_res_bgr = cv2.cvtColor(debug_res, cv2.COLOR_GRAY2BGR)
                        display_frame[10:10+debug_size[0], 170:170+debug_size[1]] = debug_res_bgr
                        cv2.putText(display_frame, "Preprocessed", (175, debug_size[0] + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Draw using last prediction
                if last_prediction is not None and last_bbox is not None:
                    draw_prediction_ui(display_frame, last_prediction, last_confidence, 
                                     last_bbox, last_all_preds)
        else:
            # No hand detected - clear buffer after a short delay
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
    cv2.imshow("ASL Digit Recognition - Improved V2", display_frame)
    
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
    elif key == ord('d') or key == ord('D'):
        show_debug = not show_debug
        print(f"{'🔍 Debug view enabled' if show_debug else '🔍 Debug view disabled'}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n✅ Session ended")
print("=" * 60)
