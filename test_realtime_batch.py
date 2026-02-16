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
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = 'asl_digit_recognition_model.keras'
IMAGE_SIZE = (128, 128)
CAPTURE_FRAMES = 60  # Capture 60 frames for batch processing
CONFIDENCE_THRESHOLD = 0.5
CAPTURE_FPS = 60  # Target FPS for capture

print("=" * 60)
print("🚀 ASL DIGIT RECOGNITION - BATCH PROCESSING MODE")
print("=" * 60)
print("\n✨ BATCH PROCESSING FEATURES:")
print(f"  • Captures {CAPTURE_FRAMES} frames rapidly")
print("  • Processes each frame like a static image")
print("  • Shows ALL frames with predictions")
print("  • Uses MAJORITY VOTING for final result")
print("  • Same preprocessing as training data")
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
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4
)
detector = vision.HandLandmarker.create_from_options(options)

def get_hand_bbox(hand_landmarks, img_width, img_height, padding=50):
    """Calculate bounding box around detected hand"""
    x_coords = [lm.x * img_width for lm in hand_landmarks]
    y_coords = [lm.y * img_height for lm in hand_landmarks]
    
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(img_width, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(img_height, int(max(y_coords)) + padding)
    
    return x_min, y_min, x_max, y_max

def preprocess_hand_image(hand_roi):
    """
    Preprocess hand region EXACTLY like training data
    """
    # Convert to grayscale
    if len(hand_roi.shape) == 3:
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = hand_roi
    
    # Resize to target size using INTER_AREA
    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    preprocessed = np.expand_dims(normalized, axis=[0, -1])
    
    return preprocessed, resized

def capture_and_process_batch(cap, detector, model):
    """
    Capture 60 frames and process them as batch
    Returns list of (preprocessed_image, prediction, confidence)
    """
    captured_frames = []
    processed_data = []
    
    print("\n🎬 Starting batch capture...")
    print(f"   Capturing {CAPTURE_FRAMES} frames...")
    
    # Capture frames rapidly
    frame_count = 0
    attempts = 0
    max_attempts = CAPTURE_FRAMES * 3  # Max attempts to get valid frames
    
    while frame_count < CAPTURE_FRAMES and attempts < max_attempts:
        attempts += 1
        success, frame = cap.read()
        if not success:
            continue
        
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detect hand
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            h, w = frame.shape[:2]
            bbox = get_hand_bbox(hand_landmarks, w, h, padding=50)
            x_min, y_min, x_max, y_max = bbox
            
            # Extract hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]
            
            if hand_roi.size > 0 and hand_roi.shape[0] > 10 and hand_roi.shape[1] > 10:
                captured_frames.append(hand_roi.copy())
                frame_count += 1
                
                # Show progress every 10 frames
                if frame_count % 10 == 0:
                    print(f"   ✓ Captured {frame_count}/{CAPTURE_FRAMES} frames")
    
    if frame_count < CAPTURE_FRAMES:
        print(f"⚠️  Warning: Only captured {frame_count} frames (target: {CAPTURE_FRAMES})")
    else:
        print(f"✅ Captured {frame_count} frames successfully!")
    
    # Process all captured frames
    print(f"\n🔄 Processing {len(captured_frames)} frames...")
    
    for idx, hand_roi in enumerate(captured_frames):
        # Preprocess exactly like training data
        preprocessed, processed_gray = preprocess_hand_image(hand_roi)
        
        # Get prediction
        predictions = model.predict(preprocessed, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        processed_data.append({
            'index': idx,
            'processed_image': processed_gray,
            'prediction': predicted_class,
            'confidence': confidence,
            'all_predictions': predictions
        })
        
        # Show progress every 10 frames
        if (idx + 1) % 10 == 0:
            print(f"   ✓ Processed {idx + 1}/{len(captured_frames)} frames")
    
    print(f"✅ Processing complete!")
    
    return processed_data

def visualize_batch_results(processed_data):
    """
    Create visualization of all processed frames with predictions
    """
    num_frames = len(processed_data)
    
    # Calculate grid size (try to make it roughly square)
    cols = int(np.ceil(np.sqrt(num_frames)))
    rows = int(np.ceil(num_frames / cols))
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    fig.suptitle(f'Batch Processing: {num_frames} Frames with Predictions', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each frame
    for idx, data in enumerate(processed_data):
        ax = axes[idx]
        
        # Display image
        ax.imshow(data['processed_image'], cmap='gray')
        
        # Color code by confidence
        if data['confidence'] >= 0.7:
            color = 'green'
        elif data['confidence'] >= 0.5:
            color = 'orange'
        else:
            color = 'red'
        
        # Title with prediction and confidence
        title = f"#{idx+1}: {data['prediction']}\n{data['confidence']:.1%}"
        ax.set_title(title, color=color, fontsize=8, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('batch_processing_results.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved: batch_processing_results.png")
    
    plt.show()

def analyze_predictions(processed_data):
    """
    Analyze predictions using majority voting and confidence
    """
    print("\n" + "=" * 60)
    print("📊 BATCH ANALYSIS RESULTS")
    print("=" * 60)
    
    # Get all predictions
    predictions = [d['prediction'] for d in processed_data]
    confidences = [d['confidence'] for d in processed_data]
    
    # Count predictions
    prediction_counts = Counter(predictions)
    
    print(f"\n📈 Prediction Distribution:")
    for digit, count in sorted(prediction_counts.items()):
        percentage = (count / len(predictions)) * 100
        bar = "█" * int(percentage / 2)
        print(f"   Digit {digit}: {count:3d} frames ({percentage:5.1f}%) {bar}")
    
    # Get majority vote
    most_common = prediction_counts.most_common(1)[0]
    majority_digit = most_common[0]
    majority_count = most_common[1]
    
    # Calculate average confidence for majority digit
    majority_confidences = [d['confidence'] for d in processed_data 
                           if d['prediction'] == majority_digit]
    avg_confidence = np.mean(majority_confidences)
    
    # Overall statistics
    print(f"\n📌 Statistics:")
    print(f"   Total frames: {len(predictions)}")
    print(f"   Average confidence: {np.mean(confidences):.1%}")
    print(f"   High confidence (>70%): {sum(1 for c in confidences if c > 0.7)} frames")
    print(f"   Medium confidence (50-70%): {sum(1 for c in confidences if 0.5 <= c <= 0.7)} frames")
    print(f"   Low confidence (<50%): {sum(1 for c in confidences if c < 0.5)} frames")
    
    print(f"\n" + "=" * 60)
    print(f"🏆 FINAL RESULT (Majority Vote)")
    print("=" * 60)
    print(f"\n   Predicted Digit: {majority_digit}")
    print(f"   Votes: {majority_count}/{len(predictions)} ({(majority_count/len(predictions)*100):.1f}%)")
    print(f"   Average Confidence: {avg_confidence:.1%}")
    print(f"\n" + "=" * 60)
    
    return majority_digit, avg_confidence

def draw_info_text(frame, text, position, color=(255, 255, 255)):
    """Draw text with background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    
    # Draw background
    cv2.rectangle(frame, (x-5, y-text_size[1]-5), 
                  (x+text_size[0]+5, y+5), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

# Initialize webcam
print("\n📹 Initializing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit(1)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

print(f"✅ Camera initialized")

print("\n" + "=" * 60)
print("INSTRUCTIONS:")
print("  1. Position your hand showing a digit (0-9)")
print("  2. Press SPACE when ready to capture")
print(f"  3. Hold steady while {CAPTURE_FRAMES} frames are captured")
print("  4. View results and visualization")
print("  5. Press ESC to exit or SPACE to capture again")
print("=" * 60 + "\n")

capturing = False
batch_mode = False

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to read from webcam")
        break
    
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    
    if not capturing:
        # Live preview mode
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            h, w = frame.shape[:2]
            bbox = get_hand_bbox(hand_landmarks, w, h, padding=50)
            x_min, y_min, x_max, y_max = bbox
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), 
                         (0, 255, 0), 3)
            
            draw_info_text(display_frame, "Hand detected - Press SPACE to capture", 
                          (10, 40), (0, 255, 0))
        else:
            draw_info_text(display_frame, "Show your hand - Press SPACE when ready", 
                          (10, 40), (0, 0, 255))
        
        # Draw instructions at bottom
        h, w = display_frame.shape[:2]
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        cv2.putText(display_frame, f"SPACE: Capture {CAPTURE_FRAMES} frames | ESC: Exit", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("ASL Batch Processing - Press SPACE to capture", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        print("\n👋 Exiting...")
        break
    elif key == 32:  # SPACE
        capturing = True
        cv2.destroyAllWindows()
        
        # Capture and process batch
        processed_data = capture_and_process_batch(cap, detector, model)
        
        if len(processed_data) > 0:
            # Analyze predictions
            final_digit, final_confidence = analyze_predictions(processed_data)
            
            # Visualize results
            print(f"\n🎨 Creating visualization...")
            visualize_batch_results(processed_data)
        else:
            print("❌ No valid frames captured. Please try again.")
        
        capturing = False
        print(f"\n Press SPACE to capture again or ESC to exit...")

# Cleanup
cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n✅ Session ended")
print("=" * 60)
