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
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
MODEL_PATH = 'asl_digit_recognition_model.keras'
IMAGE_SIZE = (128, 128)

print("=" * 70)
print("🔍 ASL DIAGNOSTIC TOOL - Visual Analysis")
print("=" * 70)
print("\nThis tool helps you see what the model sees!")
print("=" * 70)

# Load model
print("\n📦 Loading main model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Initialize MediaPipe
print("\n🖐️  Initializing hand detector...")
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

def get_hand_bbox(hand_landmarks, img_width, img_height, padding=80):
    """Calculate bounding box around detected hand"""
    x_coords = [lm.x * img_width for lm in hand_landmarks]
    y_coords = [lm.y * img_height for lm in hand_landmarks]
    
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(img_width, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(img_height, int(max(y_coords)) + padding)
    
    return x_min, y_min, x_max, y_max

def preprocess_hand_image(hand_roi):
    """Preprocess hand region exactly like training"""
    if len(hand_roi.shape) == 3:
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = hand_roi
    
    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    preprocessed = np.expand_dims(normalized, axis=[0, -1])
    
    return preprocessed, gray, resized, normalized

# Initialize webcam
print("\n📹 Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ Camera initialized")
print("\n" + "=" * 70)
print("INSTRUCTIONS:")
print("  • Show your hand with a digit sign")
print("  • Press SPACEBAR to capture and analyze")
print("  • Press ESC to exit")
print("=" * 70 + "\n")

capture_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # Detect hand
    detection_result = detector.detect(mp_image)
    
    hand_detected = False
    
    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        hand_detected = True
        
        h, w = frame.shape[:2]
        bbox = get_hand_bbox(hand_landmarks, w, h, padding=80)
        x_min, y_min, x_max, y_max = bbox
        
        hand_roi = frame[y_min:y_max, x_min:x_max]
        
        if hand_roi.size > 0 and hand_roi.shape[0] > 10 and hand_roi.shape[1] > 10:
            # Draw bounding box
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Quick prediction
            preprocessed, _, _, _ = preprocess_hand_image(hand_roi)
            predictions = model.predict(preprocessed, verbose=0)[0]
            predicted_digit = np.argmax(predictions)
            confidence = predictions[predicted_digit]
            
            # Display on frame
            cv2.putText(display_frame, f"Prediction: {predicted_digit} ({confidence:.1%})", 
                       (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Instructions
    status_text = "✓ Hand detected - Press SPACEBAR to analyze" if hand_detected else "Show hand sign..."
    color = (0, 255, 0) if hand_detected else (0, 165, 255)
    
    cv2.putText(display_frame, status_text, (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    height, width = display_frame.shape[:2]
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, height - 50), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
    
    cv2.putText(display_frame, "SPACEBAR: Capture & Analyze | ESC: Exit", 
               (width//2 - 250, height - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Diagnostic Tool", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        print("\n👋 Exiting...")
        break
    elif key == 32:  # SPACEBAR
        if hand_detected and hand_roi.size > 0:
            capture_count += 1
            print(f"\n📸 Capture #{capture_count}")
            
            # Process with all stages visible
            preprocessed, gray_roi, resized, normalized = preprocess_hand_image(hand_roi)
            predictions = model.predict(preprocessed, verbose=0)[0]
            predicted_digit = np.argmax(predictions)
            confidence = predictions[predicted_digit]
            
            # Create diagnostic visualization
            fig = plt.figure(figsize=(18, 10))
            fig.suptitle(f'Diagnostic Analysis - Capture #{capture_count}', fontsize=16, fontweight='bold')
            
            # Original ROI
            ax1 = plt.subplot(2, 4, 1)
            ax1.imshow(cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB))
            ax1.set_title('1. Original ROI\n(Cropped from camera)', fontsize=11)
            ax1.axis('off')
            
            # Grayscale
            ax2 = plt.subplot(2, 4, 2)
            ax2.imshow(gray_roi, cmap='gray')
            ax2.set_title('2. Grayscale\n(Color removed)', fontsize=11)
            ax2.axis('off')
            
            # Resized
            ax3 = plt.subplot(2, 4, 3)
            ax3.imshow(resized, cmap='gray')
            ax3.set_title(f'3. Resized to {IMAGE_SIZE}\n(Model input size)', fontsize=11)
            ax3.axis('off')
            
            # Normalized
            ax4 = plt.subplot(2, 4, 4)
            ax4.imshow(normalized, cmap='gray')
            ax4.set_title('4. Normalized [0, 1]\n(What model sees)', fontsize=11)
            ax4.axis('off')
            
            # Prediction probabilities
            ax5 = plt.subplot(2, 1, 2)
            digits = list(range(10))
            colors = ['green' if i == predicted_digit else 'skyblue' for i in digits]
            bars = ax5.bar(digits, predictions, color=colors, alpha=0.8, edgecolor='black')
            ax5.set_xlabel('Digit Class', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Confidence', fontsize=12, fontweight='bold')
            ax5.set_title(f'Model Predictions | Winner: {predicted_digit} ({confidence:.2%})', 
                         fontsize=13, fontweight='bold')
            ax5.set_ylim([0, 1.0])
            ax5.set_xticks(digits)
            ax5.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for bar, pred in zip(bars, predictions):
                height = bar.get_height()
                if height > 0.05:  # Only show if > 5%
                    ax5.text(bar.get_x() + bar.get_width()/2., height,
                            f'{pred:.1%}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Add analysis text
            analysis_text = f"""
ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Predicted Digit: {predicted_digit}
Confidence: {confidence:.2%}

Top 3 Predictions:
"""
            top_3 = np.argsort(predictions)[-3:][::-1]
            for i, idx in enumerate(top_3, 1):
                analysis_text += f"  {i}. Digit {idx}: {predictions[idx]:.2%}\n"
            
            analysis_text += f"""
ROI Shape: {hand_roi.shape}
Preprocessed Shape: {preprocessed.shape}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            # Add text box
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            fig.text(0.02, 0.02, analysis_text, fontsize=10, 
                    family='monospace', verticalalignment='bottom', bbox=props)
            
            plt.tight_layout(rect=[0, 0.12, 1, 0.96])
            
            # Save diagnostic image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnostic_capture_{capture_count}_{timestamp}_digit{predicted_digit}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {filename}")
            
            plt.show(block=False)
            plt.pause(0.1)
            
            # Print to console
            print(analysis_text)
            
            if confidence < 0.5:
                print("⚠️  LOW CONFIDENCE - Possible issues:")
                print("   • Hand position different from training data")
                print("   • Lighting conditions vary")
                print("   • Sign not clear or hand not fully visible")
                if predicted_digit in [6, 7, 8]:
                    print("   • Digits 6,7,8 are very similar - consider retraining with more data")

cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n✅ Diagnostic session ended")
print(f"📊 Total captures: {capture_count}")
print("=" * 70)
