import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import cv2
import numpy as np
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = 'asl_digit_recognition_model.keras'
TEST_IMAGES_PATH = r"dataset\Preprocessed_ASL_Digits"
IMAGE_SIZE = (128, 128)
NUM_SAMPLES_PER_CLASS = 3

print("=" * 60)
print("🧪 ASL DIGIT RECOGNITION - STATIC IMAGE TESTING")
print("=" * 60)

# Load the trained model
print("\n📦 Loading trained model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

def preprocess_image(img_path):
    """Load and preprocess image for prediction"""
    # Read grayscale image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize if needed
    if img.shape != IMAGE_SIZE:
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalize
    normalized = img.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    preprocessed = np.expand_dims(normalized, axis=[0, -1])
    
    return preprocessed, img

# Test on sample images
print("\n🔍 Testing on sample images from each class...\n")

fig, axes = plt.subplots(10, NUM_SAMPLES_PER_CLASS, figsize=(12, 20))
fig.suptitle('ASL Digit Recognition - Test Results', fontsize=16, fontweight='bold')

correct_predictions = 0
total_predictions = 0

for true_label in range(10):
    label_dir = Path(TEST_IMAGES_PATH) / str(true_label)
    
    if not label_dir.exists():
        print(f"⚠️  Warning: Directory for label {true_label} not found")
        continue
    
    # Get sample images
    image_files = list(label_dir.glob("*.jpg")) + \
                 list(label_dir.glob("*.jpeg")) + \
                 list(label_dir.glob("*.png"))
    
    # Test on first few samples
    samples = image_files[:NUM_SAMPLES_PER_CLASS]
    
    for idx, img_file in enumerate(samples):
        # Preprocess and predict
        preprocessed, original = preprocess_image(img_file)
        
        if preprocessed is None:
            continue
        
        predictions = model.predict(preprocessed, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        # Update accuracy
        total_predictions += 1
        if predicted_class == true_label:
            correct_predictions += 1
        
        # Plot
        ax = axes[true_label, idx]
        ax.imshow(original, cmap='gray')
        
        # Color-code prediction (green for correct, red for incorrect)
        color = 'green' if predicted_class == true_label else 'red'
        title = f"Pred: {predicted_class}\nConf: {confidence:.2%}"
        ax.set_title(title, color=color, fontsize=9)
        ax.axis('off')
        
        # Print result
        status = "✅" if predicted_class == true_label else "❌"
        print(f"{status} True: {true_label}, Predicted: {predicted_class}, "
              f"Confidence: {confidence:.2%}")

# Calculate accuracy
accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

print("\n" + "=" * 60)
print(f"📊 TEST RESULTS")
print("=" * 60)
print(f"Total Predictions: {total_predictions}")
print(f"Correct: {correct_predictions}")
print(f"Incorrect: {total_predictions - correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
print("=" * 60)

# Save visualization
plt.tight_layout()
plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Test results visualization saved: test_results.png")

plt.show()
