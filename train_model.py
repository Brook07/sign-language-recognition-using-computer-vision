import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Configuration
DATASET_PATH = r"dataset\Preprocessed_ASL_Digits"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25

print("=" * 60)
print("🎯 TRAINING ASL DIGIT RECOGNITION MODEL")
print("=" * 60)

# STEP 1: LOAD PREPROCESSED DATA
print("\n📂 STEP 1: Loading preprocessed dataset...")

X_data = []
y_data = []

for label in range(10):
    label_dir = Path(DATASET_PATH) / str(label)
    
    if not label_dir.exists():
        print(f"⚠️  Warning: Directory for label {label} not found")
        continue
    
    image_files = list(label_dir.glob("*.jpg")) + \
                 list(label_dir.glob("*.jpeg")) + \
                 list(label_dir.glob("*.png"))
    
    print(f"  Loading Sign {label}... ({len(image_files)} images)")
    
    for img_file in image_files:
        # Load grayscale image
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Normalize to [0, 1]
        normalized = img.astype(np.float32) / 255.0
        
        X_data.append(normalized)
        y_data.append(label)

# Convert to numpy arrays
X_data = np.array(X_data)
y_data = np.array(y_data)

print(f"\n✅ Loaded {len(X_data)} images")
print(f"   Image shape: {X_data.shape}")

# Add channel dimension for CNN
X_data = np.expand_dims(X_data, axis=-1)
print(f"   Shape after adding channel: {X_data.shape}")

# STEP 2: SPLIT DATA
print("\n📊 STEP 2: Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

print(f"   Training set: {len(X_train)} images")
print(f"   Test set: {len(X_test)} images")

# STEP 3: BUILD CNN MODEL
print("\n🏗️  STEP 3: Building CNN model...")

model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Fourth convolutional block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # 10 classes (digits 0-9)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Architecture:")
model.summary()

# STEP 4: TRAIN THE MODEL
print("\n🎯 STEP 4: Training the model...")

# Add callbacks for better training
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# STEP 5: EVALUATE
print("\n📈 STEP 5: Evaluating on test set...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'='*50}")
print(f"✨ TRAINING COMPLETE!")
print(f"{'='*50}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Per-class accuracy
print("\n📊 Per-Class Accuracy:")
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

for digit in range(10):
    mask = y_test == digit
    if mask.sum() > 0:
        digit_acc = (y_pred_classes[mask] == digit).sum() / mask.sum()
        print(f"   Digit {digit}: {digit_acc*100:.1f}%")

# STEP 6: SAVE MODEL
model_filename = 'asl_digit_recognition_model.keras'
model.save(model_filename)
print(f"\n💾 Model saved as: {model_filename}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print(f"📊 Training history saved as: training_history.png")

print("\n✅ All done! You can now run:")
print("   python test_realtime_improved_v2.py")
print("=" * 60)
