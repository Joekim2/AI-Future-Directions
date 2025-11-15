import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import os
from PIL import Image
import json

print("TensorFlow version:", tf.__version__)

# Since we're simulating in Colab, we'll use a subset of a recyclable waste dataset
# In a real scenario, you would use datasets like TrashNet or create your own

def create_synthetic_dataset():
    """Create a synthetic dataset for demonstration"""
    # Categories of recyclable items
    categories = ['plastic', 'paper', 'glass', 'metal', 'cardboard']
    
    # Create synthetic data (in real scenario, use actual images)
    num_samples = 1000
    img_height, img_width = 128, 128
    
    # Generate random images and labels
    X_train = np.random.randint(0, 255, (num_samples, img_height, img_width, 3), dtype=np.uint8)
    y_train = np.random.randint(0, len(categories), num_samples)
    
    # Split into train/validation
    split_idx = int(0.8 * num_samples)
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    return (X_train, y_train), (X_val, y_val), categories

# Create dataset
(X_train, y_train), (X_val, y_val), categories = create_synthetic_dataset()

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Categories: {categories}")

# Normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# Convert labels to categorical
y_train_categorical = keras.utils.to_categorical(y_train, len(categories))
y_val_categorical = keras.utils.to_categorical(y_val, len(categories))

# Build a lightweight CNN model suitable for Edge devices
def create_lightweight_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),
        
        # Second conv block
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),
        
        # Third conv block
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),
        
        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create model
input_shape = (128, 128, 3)
num_classes = len(categories)
model = create_lightweight_model(input_shape, num_classes)

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Train the model
print("\nTraining model...")
history = model.fit(
    X_train, y_train_categorical,
    batch_size=32,
    epochs=8,
    validation_data=(X_val, y_val_categorical),
    verbose=1
)

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_val, y_val_categorical, verbose=0)
print(f"Validation Accuracy: {test_accuracy:.4f}")
print(f"Validation Loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Convert to TensorFlow Lite
print("\nConverting to TensorFlow Lite...")

# Convert to TFLite with default optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = 'recyclable_classifier.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TensorFlow Lite model saved: {tflite_model_path}")
print(f"Model size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.2f} KB)")

# Convert with optimization for size
converter_optimized = tf.lite.TFLiteConverter.from_keras_model(model)
converter_optimized.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_optimized = converter_optimized.convert()

# Save optimized model
tflite_optimized_path = 'recyclable_classifier_optimized.tflite'
with open(tflite_optimized_path, 'wb') as f:
    f.write(tflite_model_optimized)

print(f"Optimized TensorFlow Lite model saved: {tflite_optimized_path}")
print(f"Optimized model size: {len(tflite_model_optimized)} bytes ({len(tflite_model_optimized)/1024:.2f} KB)")