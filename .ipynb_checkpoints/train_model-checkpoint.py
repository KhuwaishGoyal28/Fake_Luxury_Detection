import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Set directories
train_dir = "dataset/train"
val_dir = "dataset/val"
img_size = (224, 224)
batch_size = 32

# Check if dataset directories exist and contain images
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise ValueError("Dataset directories not found. Please check the dataset path.")

if len(os.listdir(train_dir + "/real")) == 0 or len(os.listdir(train_dir + "/fake")) == 0:
    raise ValueError("Train dataset is empty! Add images to train/real and train/fake.")

if len(os.listdir(val_dir + "/real")) == 0 or len(os.listdir(val_dir + "/fake")) == 0:
    raise ValueError("Validation dataset is empty! Add images to val/real and val/fake.")

# Data Augmentation
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_data = datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

# Load MobileNetV2 base model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base model

# Custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output_layer = Dense(1, activation="sigmoid")(x)  # Binary classification

# Define the complete model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save("luxury_brand_detector.h5")
print("Model training complete and saved!")