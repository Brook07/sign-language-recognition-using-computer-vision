import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ORIGINAL_PATH = r"dataset\American_Sign_Language_Digits_Dataset"
PREPROCESSED_PATH = r"dataset\Preprocessed_ASL_Digits"

def visualize_comparison(num_samples=5):
    """Visualize original vs preprocessed images"""
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))
    fig.suptitle('ASL Preprocessing: Original → Grayscale → Normalized (400×400)', 
                 fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        label = i  # Show signs 0-4
        
        # Get a sample image
        original_dir = Path(ORIGINAL_PATH) / str(label) / f"Input Images - Sign {label}"
        preprocessed_dir = Path(PREPROCESSED_PATH) / str(label)
        
        original_images = list(original_dir.glob("*.jpg")) + \
                         list(original_dir.glob("*.jpeg")) + \
                         list(original_dir.glob("*.png"))
        if not original_images:
            continue
        
        sample_img = original_images[0]
        
        # Find corresponding preprocessed image (same filename)
        preprocessed_images = list(preprocessed_dir.glob(sample_img.name)) + \
                             list(preprocessed_dir.glob("*.jpg")) + \
                             list(preprocessed_dir.glob("*.png"))
        if not preprocessed_images:
            continue
        preprocessed_img = preprocessed_images[0]
        
        # Read original (RGB)
        original = cv2.imread(str(sample_img))
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Read preprocessed
        preprocessed = cv2.imread(str(preprocessed_img), cv2.IMREAD_GRAYSCALE)
        
        # Create grayscale version at original size for comparison
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # Plot
        axes[i, 0].imshow(original_rgb)
        axes[i, 0].set_title(f'Original (Sign {label})\n400×400 RGB', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(original_gray, cmap='gray')
        axes[i, 1].set_title(f'Grayscale\n400×400', fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(preprocessed, cmap='gray')
        axes[i, 2].set_title(f'Preprocessed\n400×400 Normalized', fontsize=10)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved as 'preprocessing_comparison.png'")
    plt.show()

def print_statistics():
    """Print statistics about the preprocessing"""
    print("\n📊 PREPROCESSING STATISTICS")
    print("=" * 50)
    
    for label in range(10):
        preprocessed_dir = Path(PREPROCESSED_PATH) / str(label)
        if preprocessed_dir.exists():
            all_images = list(preprocessed_dir.glob("*.jpg")) + list(preprocessed_dir.glob("*.png"))
            num_images = len(all_images)
            
            # Sample one image to check properties
            if not all_images:
                continue
            sample = all_images[0]
            img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
            
            print(f"Sign {label}: {num_images} images | "
                  f"Shape: {img.shape} | "
                  f"Dtype: {img.dtype} | "
                  f"Range: [{img.min()}, {img.max()}]")
    
    print("=" * 50)
    print("\n✨ Benefits of Preprocessing:")
    print("   • 66.67% reduction in channels (3 → 1)")
    print("   • Original spatial dimensions maintained (400×400)")
    print("   • Normalized values ready for neural network training")
    print("   • Grayscale format reduces memory usage by 67%")

if __name__ == "__main__":
    print_statistics()
    print("\n📸 Creating visualization...")
    visualize_comparison()
