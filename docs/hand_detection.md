# Sign Language Recognition - Build Process

## A. Tasks Completed ✓

### 1. Hand Detection & Landmark Tracking
Successfully implemented real-time hand detection and landmark tracking using computer vision.

#### Technology Stack
- **MediaPipe**: Google's hand landmark detection model (Tasks API)
- **OpenCV**: Video capture and image processing
- **Python**: Core programming language

#### Key Components

**Hand Detection Model**
- Using MediaPipe's pre-trained `hand_landmarker.task` model
- Detects up to 1 hand per frame
- Confidence thresholds set at 0.7 for accuracy

**Landmark Detection**
- 21 key points tracked per hand
- Includes fingertips, joints, and palm points
- Real-time coordinate tracking (x, y, z)

**Visualization**
- Green lines connecting hand landmarks (skeleton)
- Red circles on each landmark point
- Live webcam feed with overlay

---

### 2. Dataset Acquisition ✓
**Dataset:** American Sign Language Digits Dataset (0-9)
- **Total Images:** 5,000 (500 per digit)
- **Format:** 400×400 RGB JPEG images
- **Source:** Pre-collected dataset in `dataset/American_Sign_Language_Digits_Dataset/`
- **Structure:** Organized by digit (0-9) with input/output image folders

---

### 3. Dataset Preprocessing ✓
Implemented comprehensive preprocessing pipeline in `preprocess_dataset.py`

#### Preprocessing Steps:
1. **RGB → Grayscale Conversion** (reduces from 3 to 1 channel)
2. **Resize:** Maintained 400×400 dimensions (optional 64×64 for smaller model)
3. **Normalization:** Pixel values scaled to [0, 1] range
4. **Saved:** Preprocessed images to `dataset/Preprocessed_ASL_Digits/`

#### Results:
- ✓ 5,000 images preprocessed successfully
- ✓ 67% memory reduction (3 channels → 1 channel)
- ✓ Normalized and ready for neural network training

---

### 4. Landmark Extraction ✓
Created `extract_landmarks.py` to extract hand landmarks from dataset images

#### Implementation:
- Used MediaPipe HandLandmarker to extract 21 landmarks (x, y, z) per image
- Processed all 5,000 images across 10 digit classes
- Saved landmarks to `asl_digits_landmarks.csv` (63 features + label)

#### Features Extracted:
- 21 landmarks × 3 coordinates = 63 features per image
- Labels: 0-9 for each corresponding digit

---

### 5. CNN Model Training ✓
Built and trained a Convolutional Neural Network in `train_cnn.py`

#### Model Architecture:
- **Input:** 128×128 grayscale images
- **Conv Blocks:** 4 blocks (32 → 64 → 128 → 256 filters)
- **Regularization:** BatchNormalization + Dropout (0.25-0.5)
- **Dense Layers:** 512 → 256 → 10 (output classes)
- **Total Parameters:** ~82.4M
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy

#### Training Configuration:
- **Train/Test Split:** 80/20 (4,000/1,000 images)
- **Validation Split:** 20% of training data
- **Batch Size:** 32
- **Epochs:** 20 (with early stopping)
- **Callbacks:** EarlyStopping, ReduceLROnPlateau

---

### 6. Model Evaluation & Accuracy ✓

#### Test Set Performance:
- **Test Accuracy:** ~96.67% (on static image testing)
- **Confidence Scores:** Most predictions 99%+ confidence
- **Confusion Matrix:** Generated and saved

#### Per-Class Performance:
- Signs 0, 1, 3, 5, 6, 7, 8: 100% accuracy
- Sign 2: 99.86% average confidence
- Sign 4: 52-99% confidence range (some confusion)
- Sign 9: One misclassification as 8 (29/30 correct)

#### Outputs Generated:
- ✓ `asl_digit_recognition_model.keras` - Trained model file
- ✓ `training_history.png` - Training/validation curves
- ✓ `confusion_matrix.png` - Visual confusion matrix
- ✓ `test_results.png` - Static test results visualization

---

### 7. Real-Time Testing ✓
Implemented two testing approaches:

#### A. Real-Time Webcam Testing (`test_realtime.py`)
- **Features:**
  - Live webcam feed with hand detection
  - Automatic bounding box around detected hand
  - Real-time digit prediction (0-9)
  - Confidence score display
  - Color-coded predictions (green: high confidence, orange: low)
  
- **Controls:**
  - ESC: Exit
  - SPACE: Pause/Resume
  - C: Clear predictions

#### B. Static Image Testing (`test_static.py`)
- Tests on preprocessed dataset samples
- Generates visualization grid (10 classes × 3 samples)
- Color-coded results (green: correct, red: incorrect)
- **Result:** 96.67% accuracy (29/30 correct predictions)

---

## B. Tasks To Be Done

### 1. Fine-Tuning & Optimization ⏳
- [ ] Data augmentation (rotation, scaling, brightness)
- [ ] Hyperparameter tuning (learning rate, batch size)
- [ ] Model architecture optimization (reduce parameters)
- [ ] Cross-validation for robust evaluation
- [ ] Address Sign 4 confusion (lower confidence issue)

### 2. Expand Functionality ⏳
- [ ] Add A-Z alphabet recognition (beyond digits 0-9)
- [ ] Multi-hand detection support
- [ ] Dynamic gesture recognition (temporal sequences)
- [ ] Custom sign language support

### 3. Deployment 🎯
- [ ] FastAPI backend for model serving
- [ ] React frontend for web interface
- [ ] Dockerize application
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] API endpoint for predictions
- [ ] Real-time streaming support

### 4. Production Features ⏳
- [ ] Model versioning and monitoring
- [ ] Error handling and logging
- [ ] Performance optimization (inference speed)
- [ ] Mobile app development
- [ ] Batch prediction support
- [ ] User authentication and usage tracking

---

## Project Structure

```
sign-language-recognition-using-computer-vision/
├── hand_detection.py           # Real-time hand landmark detection
├── extract_landmarks.py        # Extract landmarks from dataset
├── preprocess_dataset.py       # Image preprocessing pipeline
├── train_cnn.py               # CNN model training
├── test_realtime.py           # Real-time webcam testing
├── test_static.py             # Static image testing
├── visualize_preprocessing.py # Preprocessing visualization
├── hand_landmarker.task       # MediaPipe model file
├── asl_digit_recognition_model.keras  # Trained model
├── asl_digits_landmarks.csv   # Extracted landmarks
├── dataset/
│   ├── American_Sign_Language_Digits_Dataset/
│   └── Preprocessed_ASL_Digits/
└── docs/
    └── hand_detection.md      # This documentation
```

---

## Quick Start Guide

### 1. Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install tensorflow opencv-python mediapipe scikit-learn matplotlib pandas numpy
```

### 2. Preprocess Dataset
```bash
python preprocess_dataset.py
```

### 3. Train Model
```bash
python train_cnn.py
```

### 4. Test Real-Time
```bash
python test_realtime.py
```

### 5. Test Static Images
```bash
python test_static.py
```

---

*Last Updated: February 14, 2026*
