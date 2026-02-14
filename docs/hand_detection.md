# Sign Language Recognition - Build Process

## Current Progress: Hand Detection ✓

### What I Built

Successfully implemented real-time hand detection and landmark tracking using computer vision.

### How I Built It

#### 1. **Technology Stack**
- **MediaPipe**: Google's hand landmark detection model
- **OpenCV**: Video capture and image processing
- **Python**: Core programming language

#### 2. **Key Components**

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

#### 3. **Implementation Details**

**Configuration**
```python
num_hands=1
min_hand_detection_confidence=0.7
min_hand_presence_confidence=0.7
min_tracking_confidence=0.7
```

**Hand Connections Mapped**
- Thumb: 4 segments
- Index, Middle, Ring, Pinky: 4 segments each
- Palm: connecting lines between fingers

**Frame Processing**
1. Capture frame from webcam
2. Flip horizontally (mirror effect)
3. Convert BGR → RGB
4. Run MediaPipe detection
5. Draw landmarks and connections
6. Display result

#### 4. **Features Working**
- ✓ Real-time hand detection
- ✓ 21-point landmark tracking
- ✓ Visual feedback (boxes/skeleton on hand)
- ✓ Webcam integration
- ✓ ESC key to exit

### Next Steps
- [ ] Gesture recognition
- [ ] Sign language classification
- [ ] Training custom model
- [ ] Multi-hand support

---
*Last Updated: February 13, 2026*
