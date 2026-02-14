# Sign Language Recognition using Computer Vision

A real-time ASL digit recognition system (0-9) using deep learning and computer vision. This project uses MediaPipe for hand detection and a custom CNN for digit classification.

## 🎯 Current Status: Trained & Tested ✓

**What this project does:**
- Real-time hand detection and tracking using MediaPipe
- Recognizes ASL digits (0-9) with 96.67% accuracy
- Live webcam predictions with confidence scores
- Comprehensive preprocessing and training pipeline

**Key Features:**
- ✅ Real-time hand landmark detection (21 points)
- ✅ CNN model trained on 5,000 images
- ✅ Live webcam testing with bounding boxes
- ✅ Static image evaluation
- ✅ 96.67% test accuracy
- ⏳ Deployment (coming soon)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam (for real-time testing)

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

### Usage

#### 1. Preprocess Dataset (if needed)
```bash
python preprocess_dataset.py
```

#### 2. Train Model (if needed)
```bash
python train_cnn.py
```

#### 3. Test Real-Time (Recommended)
```bash
python test_realtime.py
```
- Show ASL digit signs (0-9) to your webcam
- Press ESC to exit, SPACE to pause, C to clear

#### 4. Test on Static Images
```bash
python test_static.py
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 96.67% |
| Training Images | 4,000 |
| Test Images | 1,000 |
| Classes | 10 (digits 0-9) |
| Model Size | 314 MB |
| Parameters | 82.4M |

**Per-Class Accuracy:**
- Signs 0, 1, 3, 5, 6, 7, 8: ~100%
- Sign 2: 99.86%
- Sign 4: 70-99% (some variation)
- Sign 9: 96.67% (1 misclassification)

## 🏗️ Technology Stack

- **Deep Learning:** TensorFlow/Keras
- **Computer Vision:** OpenCV, MediaPipe
- **Data Processing:** NumPy, pandas, scikit-learn
- **Visualization:** Matplotlib

## 📁 Project Structure

```
sign-language-recognition-using-computer-vision/
├── hand_detection.py              # Real-time hand tracking
├── extract_landmarks.py           # Extract hand landmarks from images
├── preprocess_dataset.py          # Image preprocessing pipeline
├── train_cnn.py                   # CNN training script
├── test_realtime.py              # Real-time webcam testing
├── test_static.py                # Static image testing
├── visualize_preprocessing.py    # Visualize preprocessing results
├── hand_landmarker.task          # MediaPipe model file
├── asl_digit_recognition_model.keras  # Trained CNN model
├── dataset/
│   ├── American_Sign_Language_Digits_Dataset/  # Original images
│   └── Preprocessed_ASL_Digits/               # Preprocessed images
├── docs/
│   └── hand_detection.md         # Detailed build documentation
└── README.md                      # This file
```

## 🔧 How It Works

### 1. Hand Detection
- Uses MediaPipe's HandLandmarker for detecting hands in real-time
- Tracks 21 key points per hand
- Extracts bounding box around detected hand

### 2. Preprocessing
- Converts RGB images to grayscale
- Resizes to 128×128 pixels
- Normalizes pixel values to [0, 1]

### 3. CNN Architecture
```
Input (128×128×1)
↓
Conv2D (32) → BatchNorm → MaxPool → Dropout
Conv2D (64) → BatchNorm → MaxPool → Dropout
Conv2D (128) → BatchNorm → MaxPool → Dropout
Conv2D (256) → BatchNorm → MaxPool → Dropout
↓
Flatten
Dense (512) → BatchNorm → Dropout
Dense (256) → BatchNorm → Dropout
Dense (10, softmax)
```

### 4. Real-Time Prediction
- Detects hand in webcam frame
- Crops hand region
- Preprocesses and feeds to CNN
- Displays prediction with confidence score

## 📈 Training Details

- **Dataset:** 5,000 ASL digit images (500 per class)
- **Train/Val/Test Split:** 64/16/20
- **Batch Size:** 32
- **Epochs:** 20 (with early stopping)
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau

## 🎬 Generated Outputs

- `asl_digit_recognition_model.keras` - Trained model
- `training_history.png` - Training curves
- `confusion_matrix.png` - Confusion matrix
- `test_results.png` - Test predictions visualization
- `preprocessing_comparison.png` - Before/after preprocessing

## 🔮 Future Enhancements

- [ ] FastAPI backend for model serving
- [ ] React web interface
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Expand to A-Z alphabet recognition
- [ ] Multi-hand support
- [ ] Mobile app development
- [ ] Data augmentation for improved accuracy

## 📚 Documentation

For detailed build process and technical documentation, see [docs/hand_detection.md](docs/hand_detection.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MediaPipe by Google for hand detection
- TensorFlow team for the deep learning framework
- ASL dataset contributors

---

**Last Updated:** February 14, 2026  
**Status:** Training Complete, Deployment Pending

