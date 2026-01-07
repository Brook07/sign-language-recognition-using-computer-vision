# Sign Language Recognition using Computer Vision

A simple, friendly project that converts hand signs into text (and optional speech) using computer vision and a trained AI model. This README explains what the project does, why it matters, and how to get started — in plain terms.

**What this project does**
- Takes images or a live webcam feed of hand signs.
- Recognizes ASL alphabet letters (A–Z) and, later, selected words or dynamic gestures.
- Outputs the recognized letter/word and a confidence score. Can also speak the result.

**Why it matters**
- Helps deaf and hard-of-hearing people communicate more easily.
- Useful in schools, hospitals, public services, and accessibility tools.

**How it works (simple flow)**
1. Frontend captures camera frames (React + webcam).
2. Frames are sent to a FastAPI backend.
3. Backend runs a CNN model (TensorFlow/PyTorch) to predict the sign.
4. Prediction is returned to the frontend as text (and optional audio).

**Tech stack (easy list)**
- AI / ML: Python, TensorFlow or PyTorch, OpenCV, MediaPipe (hand landmarks)
- Backend: FastAPI (REST API for predictions)
- Frontend: React (webcam capture, live predictions)

**Dataset & training (practical plan)**
- Start with ASL alphabet dataset (public datasets like Kaggle). Optionally record extra examples.
- Phase 1 (Beginner): Train a CNN to recognize 26 ASL letters.
	- Dataset size suggestion: tens of thousands of images (more data → better model).
	- Training time rough estimates: CPU 4–7 hours, GPU 1–2 hours (depends on hardware).
- Phase 2 (Advanced): Add words and dynamic gestures using CNN + LSTM and/or MediaPipe landmarks.

**What you’ll learn (short list)**
- Image preprocessing and augmentation
- Building and training a CNN
- Handling overfitting and saving models
- Serving the model via an API and integrating with React
- Measuring inference latency and improving performance

**Team roles (example split)**
- Model lead: design & train models, backend API, optimization
- Frontend lead: React UI, webcam integration, demo and testing

**Final deliverables (for a strong portfolio)**
- Trained model file(s) and training logs
- FastAPI backend serving predictions
- React web app with live demo
- README (this file) + demo video

**Quick start (local, simple)**
1. Install Python 3.8+ and Node.js (for frontend).
2. Create a virtual environment and install Python dependencies (example):

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. Start the backend API (example):

```bash
uvicorn app.main:app --reload
```

4. Run the React frontend and open the demo in your browser.

(Note: add `requirements.txt` and frontend files to this repo as next steps.)

**Next steps — recommended**
- Add `requirements.txt` with core packages (TensorFlow/PyTorch, FastAPI, OpenCV, MediaPipe).
- Add a minimal FastAPI `app` and a tiny React demo to test the full pipeline.
- Collect a small custom dataset for testing and log training runs.

If you want, I can: add a `requirements.txt`, scaffold a minimal FastAPI endpoint, or create a tiny React demo to show live predictions. Tell me which one you prefer.