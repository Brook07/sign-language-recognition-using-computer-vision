import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
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
PREPROCESSED_PATH = r"dataset\Preprocessed_ASL_Digits"
IMAGE_SIZE = (128, 128)  # Auto-detected from preprocessed images
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 50
RANDOM_STATE = 42


print("=" * 60)
print("🚀 ASL DIGIT RECOGNITION - CNN TRAINING")
print("=" * 60)

# ============================================================================
# STEP 1: LOAD DATASET AND CREATE X, y
# ============================================================================
print("\n📂 STEP 1: Loading preprocessed dataset...")

X = []  # Features (images)
y = []  # Labels (digits 0-9)

for label in range(NUM_CLASSES):
    label_dir = Path(PREPROCESSED_PATH) / str(label)
    
    if not label_dir.exists():
        print(f"⚠️  Warning: Directory for label {label} not found")
        continue
    
    # Get all image files
    image_files = list(label_dir.glob("*.jpg")) + \
                 list(label_dir.glob("*.jpeg")) + \
                 list(label_dir.glob("*.png"))
    
    print(f"   Loading Sign {label}: {len(image_files)} images", end=" ")
    
    for img_file in image_files:
        # Read grayscale image
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Normalize to [0, 1]
        img_normalized = img.astype(np.float32) / 255.0
        
        X.append(img_normalized)
        y.append(label)
    
    print("✅")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Add channel dimension (height, width, channels)
X = np.expand_dims(X, axis=-1)

print(f"\n✅ Dataset loaded successfully!")
print(f"   X shape: {X.shape} (samples, height, width, channels)")
print(f"   y shape: {y.shape} (samples,)")
print(f"   Total samples: {len(X)}")
print(f"   Classes: {NUM_CLASSES} (digits 0-9)")

# ============================================================================
# STEP 2: SPLIT DATA INTO TRAIN/TEST SETS
# ============================================================================
print("\n📊 STEP 2: Splitting data into train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE,
    stratify=y  # Ensure balanced split
)

print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")
print(f"   Split ratio: 80/20")

# ============================================================================
# STEP 3: BUILD CNN MODEL
# ============================================================================
print("\n🏗️  STEP 3: Building CNN architecture...")

model = models.Sequential([
    # Input layer
    layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Fourth Convolutional Block
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    # Output layer
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Model Architecture:")
model.summary()

# ============================================================================
# STEP 4: TRAIN THE MODEL
# ============================================================================
print("\n🎯 STEP 4: Training the model...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Optimizer: Adam")
print(f"   Loss: Sparse Categorical Crossentropy\n")

# Early stopping and model checkpoint callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# STEP 5: EVALUATE THE MODEL
# ============================================================================
print("\n📈 STEP 5: Evaluating model performance...")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'=' * 60}")
print(f"✨ TRAINING COMPLETE!")
print(f"{'=' * 60}")
print(f"📊 Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📊 Test Loss: {test_loss:.4f}")
print(f"{'=' * 60}")

# Get predictions for confusion analysis
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate per-class accuracy
from sklearn.metrics import classification_report, confusion_matrix

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, 
                          target_names=[f"Sign {i}" for i in range(NUM_CLASSES)]))

# ============================================================================
# STEP 6: SAVE MODEL AND VISUALIZATIONS
# ============================================================================
print("\n💾 Saving model and visualizations...")

# Save the model
model.save('asl_digit_recognition_model.keras')
print(f"   ✅ Model saved: asl_digit_recognition_model.keras")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Training history saved: training_history.png")

# Plot confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                               display_labels=[f"Sign {i}" for i in range(NUM_CLASSES)])
disp.plot(ax=ax, cmap='Blues', colorbar=True)
ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Confusion matrix saved: confusion_matrix.png")

print(f"\n{'=' * 60}")
print(f"✅ ALL DONE! Model is ready for inference.")
print(f"{'=' * 60}")
