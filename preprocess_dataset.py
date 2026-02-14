import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import cv2
import numpy as np
from pathlib import Path

# Configuration
INPUT_DATASET_PATH = r"dataset\American_Sign_Language_Digits_Dataset"
OUTPUT_DATASET_PATH = r"dataset\Preprocessed_ASL_Digits"
TARGET_SIZE = (128, 128)  # resize to 128x128 for better performance while retaining enough detail

def preprocess_image(image_path):
    """
    Preprocess a single image:
    1. Read image
    2. Convert RGB to Grayscale
    3. Resize to target size
    4. Normalize pixel values to [0, 1]
    
    Returns preprocessed image
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized

def save_preprocessed_image(image, output_path):
    """Save normalized image (convert back to 0-255 range for saving)"""
    # Convert back to 0-255 for saving
    img_uint8 = (image * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), img_uint8)

def preprocess_dataset():
    """Preprocess entire ASL Digits dataset"""
    
    # Create output directory
    Path(OUTPUT_DATASET_PATH).mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    processed_images = 0
    skipped_images = 0
    
    print(f"🎯 Starting preprocessing...")
    print(f"   Original size: 400x400 RGB")
    print(f"   Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} Grayscale")
    print(f"   Normalization: [0, 1]\n")
    
    # Process each digit (0-9)
    for label in range(10):
        label_str = str(label)
        input_label_path = Path(INPUT_DATASET_PATH) / label_str / f"Input Images - Sign {label_str}"
        
        if not input_label_path.exists():
            print(f"⚠️  Skipping sign {label_str} - directory not found")
            continue
        
        # Create output directory for this label
        output_label_path = Path(OUTPUT_DATASET_PATH) / label_str
        output_label_path.mkdir(parents=True, exist_ok=True)
        
        # Get all images in this label folder
        image_files = list(input_label_path.glob("*.jpg")) + \
                     list(input_label_path.glob("*.jpeg")) + \
                     list(input_label_path.glob("*.png"))
        
        print(f"📁 Processing Sign {label_str}... ({len(image_files)} images)")
        
        # Process each image
        for img_file in image_files:
            total_images += 1
            
            # Preprocess
            preprocessed = preprocess_image(img_file)
            
            if preprocessed is None:
                skipped_images += 1
                continue
            
            # Save preprocessed image
            output_file = output_label_path / img_file.name
            save_preprocessed_image(preprocessed, output_file)
            
            processed_images += 1
        
        print(f"   ✅ Completed: {processed_images} images processed")
    
    print(f"\n{'='*50}")
    print(f"✨ PREPROCESSING COMPLETE!")
    print(f"{'='*50}")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Skipped (errors): {skipped_images}")
    print(f"\n📂 Preprocessed images saved to: {OUTPUT_DATASET_PATH}")
    print(f"   - Format: Grayscale")
    print(f"   - Size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"   - Pixel range: [0-255] (normalized values saved as uint8)")

if __name__ == "__main__":
    preprocess_dataset()
