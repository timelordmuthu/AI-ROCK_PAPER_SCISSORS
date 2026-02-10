import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# ==========================================
# 1. PRE-FLIGHT CHECK & CLEANING
# ==========================================
DATA_DIR = r'C:\Users\timel\Music\muthu vit\codings\mozilla firefox\aiml task\prs' # <--- CHANGE THIS

valid_classes = ['rock', 'paper', 'scissors']
found_folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]

print(f"🔍 Found folders: {found_folders}")

# Ensure we only have the 3 required folders
for folder in found_folders:
    if folder.lower() not in valid_classes:
        print(f"⚠️ Warning: Found unexpected folder '{folder}'. Please remove it.")
    else:
        # Clean hidden files inside the valid folders
        folder_path = os.path.join(DATA_DIR, folder)
        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(folder_path, file)
                print(f"🗑️ Removing junk file: {file}")
                os.remove(file_path)

# ==========================================
# 2. THE DATA LOADER (MobileNetV2 optimized)
# ==========================================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ==========================================
# 3. TRANSFER LEARNING (Building the Brain)
# ==========================================
# Using MobileNetV2 - a high-performance lightweight model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze the pre-trained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax') # 3 classes: Rock, Paper, Scissors
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 4. START TRAINING
# ==========================================
print("\n🚀 Training started...")
model.fit(train_data, validation_data=val_data, epochs=5)

# Save the final "Brain" file
model.save("rps_mobile_model.h5")
print("\n✅ Brain file 'rps_mobile_model.h5' has been created successfully!")