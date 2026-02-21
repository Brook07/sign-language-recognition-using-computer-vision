# Sign Language Recognition using Computer Vision
## � Demo

Real-time ASL digit recognition in action:

<table>
  <tr>
    <td width="50%">
      <img src="docs/demo/Screenshot 2026-02-21 200340.png" alt="Demo 1" width="100%"/>
    </td>
    <td width="50%">
      <img src="docs/demo/Screenshot 2026-02-21 200400.png" alt="Demo 2" width="100%"/>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="docs/demo/Screenshot 2026-02-21 200419.png" alt="Demo 3" width="100%"/>
    </td>
    <td width="50%">
      <img src="docs/demo/Screenshot 2026-02-21 200544.png" alt="Demo 4" width="100%"/>
    </td>
  </tr>
</table>

*Real-time predictions with confidence scores, bounding boxes, and temporal smoothing*

A real-time ASL digit recognition system (0-9) using deep learning and computer vision. This project uses MediaPipe for hand detection and a custom CNN for digit classification with temporal smoothing for stable predictions.

## 🎯 Current Status: In Development 🚧

**What this project does:**
- Real-time hand detection and tracking using MediaPipe
- Recognizes ASL digits with **99.75% validation accuracy** (training set)
- Live webcam predictions with temporal smoothing (7-frame buffer)
- Comprehensive preprocessing and training pipeline
- Advanced diagnostic tools for model analysis

**Key Features:**
- ✅ Real-time hand landmark detection (21 points)
- ✅ CNN model trained on 4,000 images per class
- ✅ **Temporal smoothing** for stable predictions
- ✅ Diagnostic tool for visual analysis
- ✅ **Working digits: 1, 2, 3, 4, 5, 9**
- ⚠️ **Known issues: Digits 6, 7, 8 need improvement**
- ⚠️ Validation accuracy != real-world performance (data diversity needed)



## �🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam (for real-time testing)
- Windows/Linux/MacOS

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd sign-language-recognition-using-computer-vision
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install tensorflow opencv-python mediapipe scikit-learn matplotlib pandas numpy
```

### 🎥 Run the Demo (1-Minute Quickstart)
```bash
python test_realtime_improved_v2.py
```
That's it! Show ASL digit signs to your webcam and see real-time predictions.

**Note:** Works best for digits 1, 2, 3, 4, 5, 9. Digits 6, 7, 8 need improvement due to training data limitations.

### Usage

#### 1. Preprocess Dataset (if needed)
```bash
python preprocess_dataset.py
```
Converts original ASL images to grayscale, resizes to 128×128, and saves preprocessed data.

#### 2. Train Model (if needed)
```bash
python train_model.py
```
Trains the main CNN model with early stopping and learning rate reduction. Achieves ~99.75% validation accuracy.

#### 3. **Test Real-Time (Main Testing)**
```bash
python test_realtime_improved_v2.py
```
- Show ASL digit signs to your webcam
- Real-time continuous predictions
- **Temporal smoothing** for stable results (7-frame buffer)
- Confidence scores with color-coded status
- Press 'R' to reset buffer, ESC to exit
- **Working best for digits: 1, 2, 3, 4, 5, 9**

#### 4. Diagnostic Tool (Debug & Analyze)
```bash
python diagnostic_tool.py
```
- Visual breakdown of preprocessing steps
- Press SPACEBAR to capture and analyze
- Shows what the model "sees"
- Identifies prediction issues
- Per-class probability visualization
- Useful for debugging digits 6, 7, 8

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **99.75%** (on training dataset) |
| **Real-World Performance** | **~70%** (generalization gap) |
| Training Images | 4,000 per class |
| Total Dataset | 40,000 images |
| Classes | 10 (digits 0-9) |
| Model Size | ~5.8 MB |
| Parameters | 1.45M |
| Input Size | 128×128 grayscale |

**Training Improvements:**
- ✅ Early Stopping (patience: 7 epochs)
- ✅ Learning Rate Reduction (patience: 4 epochs, factor: 0.5)
- ✅ Batch Normalization for stability
- ✅ Dropout (0.3-0.5) to prevent overfitting
- ✅ 20% validation split for monitoring

**Real-World Performance:**
- ✅ **Working well:** Digits 1, 2, 3, 4, 5, 9
- ⚠️ **Challenging:** Digits 6, 7, 8 (similar hand poses in ASL)
- ⚠️ **Issue:** Training data lacks diversity (same person, lighting, background)
- 🔍 **Gap:** High validation accuracy ≠ high real-world accuracy

**Real-Time Features:**
- Temporal smoothing (7-frame moving average)
- Dynamic bounding box with 80px padding
- Confidence-based color coding (GREEN: >70%, YELLOW: 50-70%, ORANGE: 30-50%, RED: <30%)
- Top-3 prediction display

## 🎯 Technical Highlights

### Model Optimization
- **98% Parameter Reduction:** From 82.4M → 1.45M parameters
- **File Size:** From 314 MB → 5.8 MB
- **Validation Accuracy:** 99.75% (on training distribution)

### Real-Time Performance
- **Frame Processing:** ~30 FPS on standard webcam
- **Latency:** < 50ms per prediction
- **Stability:** 7-frame temporal smoothing eliminates jitter

### Preprocessing Pipeline
- **Exact Match:** Train/test preprocessing identical (critical for accuracy)
- **Efficient:** Grayscale conversion reduces computation 3x
- **Robust:** 80px padding handles various hand sizes

### Advanced Features
- **Diagnostic Mode:** Visual preprocessing analysis tool
- **Temporal Smoothing:** 7-frame buffer for stable predictions
- **Confidence Display:** Color-coded confidence levels

## ⚠️ Known Limitations

### Current Issues
1. **Digits 6, 7, 8 Recognition:** Poor performance on these similar hand gestures
   - Root cause: ASL digits 6, 7, 8 have very similar hand positions
   - Training data lacks diversity (single person, uniform lighting/background)
   
2. **Generalization Gap:** 99.75% validation accuracy but ~70% real-world accuracy
   - Model overfits to training data characteristics
   - Struggles with different users, lighting, or camera angles

3. **Data Diversity:** All training images from same source
   - Same person's hand
   - Same lighting conditions
   - Same background
   - Same camera angle

### Recommended Solutions
1. **Collect Diverse Data:** Photos from multiple people, various lighting/backgrounds
2. **Data Augmentation:** Add rotation, brightness, contrast variations
3. **Hand Landmarks:** Use MediaPipe landmark positions instead of raw pixels
4. **More Training Data:** Especially for digits 6, 7, 8

## 🏗️ Technology Stack

- **Deep Learning:** TensorFlow/Keras
- **Computer Vision:** OpenCV, MediaPipe
- **Data Processing:** NumPy, pandas, scikit-learn
- **Visualization:** Matplotlib

## 📁 Project Structure

```
sign-language-recognition-using-computer-vision/
├── Core Scripts
│   ├── hand_detection.py              # Real-time hand tracking
│   ├── extract_landmarks.py           # Extract hand landmarks from images
│   ├── preprocess_dataset.py          # Image preprocessing pipeline
│   
├── Training Scripts
│   ├── train_model.py                 # Main CNN training (10 digits)
│   
├── Testing Scripts
│   ├── test_realtime_improved_v2.py   # Real-time with temporal smoothing (MAIN)
│   ├── diagnostic_tool.py             # Visual analysis & debugging tool
│   
├── Models
│   ├── hand_landmarker.task           # MediaPipe model file
│   ├── asl_digit_recognition_model.keras  # Main trained model
│   ├── asl_model_678.keras            # Experimental model for 6,7,8
│   ├── training_history.png           # Training curves visualization
│   
├── Dataset
│   ├── American_Sign_Language_Digits_Dataset/  # Original images
│   │   ├── 0/ ... 9/                  # Raw images per digit
│   └── Preprocessed_ASL_Digits/       # Preprocessed 128x128 grayscale
│       ├── 0/ ... 9/                  # Processed images per digit
│   
├── Documentation
│   ├── docs/
│   │   ├── hand_detection.md          # Detailed build documentation
│   │   └── MODEL_IMPROVEMENTS.md      # Model iteration history
│   ├── sign_language_images_for_learning/  # Reference images
│   └── README.md                       # This file
```

## 🔧 How It Works

### 1. Hand Detection
- Uses MediaPipe's HandLandmarker for detecting hands in real-time
- Tracks 21 key points per hand with high accuracy
- Extracts bounding box around detected hand (80px padding)
- Works with single hand detection for optimal performance

### 2. Preprocessing Pipeline
- **Step 1:** Converts RGB images to grayscale
- **Step 2:** Resizes to 128×128 pixels (INTER_AREA interpolation)
- **Step 3:** Normalizes pixel values to [0, 1]
- **Step 4:** Expands dimensions for model input (1, 128, 128, 1)
- **Critical:** No CLAHE or histogram equalization (to match training)

### 3. CNN Architecture
```
Input (128×128×1 grayscale)
↓
Conv2D (32 filters, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.3)
Conv2D (64 filters, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.3)
Conv2D (128 filters, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.4)
Conv2D (256 filters, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.5)
↓
Flatten (2,304 features)
Dense (512) → BatchNorm → ReLU → Dropout(0.5)
Dense (256) → BatchNorm → ReLU → Dropout(0.4)
Dense (10, softmax)
↓
Output: 10 classes (digits 0-9)
```

**Total Parameters:** 1.45M (Trainable: 1.45M)

### 4. Real-Time Prediction with Temporal Smoothing
- Detects hand in webcam frame using MediaPipe
- Crops hand region with dynamic bounding box
- Preprocesses and feeds to CNN
- **Temporal Smoothing:** Averages predictions over last 7 frames
- Displays prediction with confidence score and color coding
- Shows top-3 predictions for transparency

## 📈 Training Details

- **Dataset:** 40,000 total ASL digit images (4,000 per class)
- **Train/Validation Split:** 80/20 (32,000 train, 8,000 validation)
- **Batch Size:** 32
- **Epochs:** Up to 50 (with early stopping)
- **Optimizer:** Adam (adaptive learning rate)
- **Loss Function:** Sparse Categorical Crossentropy
- **Callbacks:** 
  - **EarlyStopping:** Patience 7, monitors validation loss
  - **ReduceLROnPlateau:** Patience 4, reduces LR by 0.5 when loss plateaus
- **Regularization:** Dropout (0.3-0.5), Batch Normalization
- **Final Validation Accuracy:** 99.75% (on validation set from same distribution)

### Model Improvements Journey
1. **Initial Model:** 96.67% accuracy, 82.4M parameters
2. **Preprocessing Fix:** Removed CLAHE mismatch between train/test
3. **Architecture Optimization:** Reduced to 1.45M parameters (98% reduction)
4. **Added Callbacks:** Early stopping and learning rate reduction
5. **Temporal Smoothing:** 7-frame buffer for stable real-time predictions
6. **Current Challenge:** Overfitting to training distribution - high validation accuracy but poor real-world generalization for digits 6,7,8

## 🎬 Generated Outputs

### Models
- `asl_digit_recognition_model.keras` - Main trained model
- `asl_model_678.keras` - Experimental model (not in use)

### Visualizations
- `training_history.png` - Training/validation curves
- `diagnostic_capture_*.png` - Visual diagnostic analysis images (from diagnostic tool)

### Real-Time Output
- Bounding boxes with confidence scores
- Top-3 predictions overlay
- Color-coded confidence levels
- Temporal smoothing buffer status

## 🔮 Future Enhancements

### Completed ✅
- [x] Real-time continuous predictions
- [x] Temporal smoothing for stability
- [x] Diagnostic tool for debugging
- [x] Model optimization (98% parameter reduction)
- [x] Preprocessing pipeline refinement

### High Priority 🔴 (Critical for Production)
- [ ] **Fix digits 6, 7, 8 recognition** (currently not working)
- [ ] **Collect diverse training data** (multiple people, lighting, backgrounds, angles)
- [ ] Implement data augmentation (rotation, brightness, contrast, perspective)
- [ ] Test with different users and environments
- [ ] Consider landmark-based approach instead of pixel-based

### Planned 🎯 (Future Features)
- [ ] Ensemble model approach for improved accuracy
- [ ] FastAPI backend for model serving
- [ ] React web interface
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Expand to A-Z alphabet recognition
- [ ] Multi-hand support
- [ ] Mobile app development (TensorFlow Lite)

## 📚 Documentation

For detailed build process and technical documentation, see:
- [docs/hand_detection.md](docs/hand_detection.md) - Hand detection implementation
- [docs/MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md) - Model iteration history

## 💡 Lessons Learned

### Key Insights
1. **Preprocessing Consistency is Critical:** Training and testing preprocessing must match EXACTLY (learned after CLAHE mismatch debugging)
2. **Temporal Smoothing Helps:** 7-frame averaging significantly stabilizes real-time predictions
3. **Data Diversity Matters Most:** Uniformity in training data (same person/lighting) severely limits real-world generalization
4. **Validation Accuracy ≠ Real Performance:** 99.75% validation but ~70% real-world shows overfitting to training distribution
5. **Model Size != Accuracy:** Reduced from 82.4M to 1.45M parameters (98% reduction) without losing performance
6. **Early Stopping Saves Time:** Prevents overfitting and unnecessary training epochs
7. **Similar Gestures are Hard:** Digits 6, 7, 8 are visually similar in ASL and require more diverse, high-quality training data
8. **Pixel-Based Has Limits:** Raw image approach struggles with variation; landmark-based may be better

### Debugging Wins
- ✅ Fixed "predicting 8 for everything" → preprocessing mismatch
- ✅ Fixed low confidence → removed threshold blocking
- ✅ Improved stability → added temporal smoothing
- ✅ Created diagnostic tool → visual debugging for preprocessing pipeline

### Current Challenges
- ⚠️ **Overfitting:** High validation accuracy masks poor real-world performance
- ⚠️ **Data Quality:** Training data too uniform (single source)
- ⚠️ **Digits 6,7,8:** Still unreliable despite multiple approaches
- 🔍 **Next Step:** Need to collect diverse real-world data or switch to landmark-based features

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MediaPipe by Google for hand detection
- TensorFlow team for the deep learning framework
- ASL dataset contributors

---

**Last Updated:** February 21, 2026  
**Status:** ⚠️ In Development - Known Issues with Digits 6,7,8  
**Model Version:** v2.0 (99.75% validation accuracy, ~70% real-world accuracy)  
**Working Digits:** 1, 2, 3, 4, 5, 9 ✅ | **Problematic:** 6, 7, 8 ❌

