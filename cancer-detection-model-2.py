import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split

# Define paths and parameters
train_images_path = 'histopathologic-cancer-detection/train'
labels_path = 'histopathologic-cancer-detection/train_labels.csv'
test_images_path = 'histopathologic-cancer-detection/test'
image_dim = (64, 64)  # Image dimensions
batch_size = 32

# Load and preprocess data
def load_image_data(image_directory, labels_path, img_size):
    labels_df = pd.read_csv(labels_path)
    image_ids = labels_df['id'].tolist()
    labels = labels_df['label'].tolist()
    
    images = []
    labels_list = []
    for img_id, label in zip(image_ids, labels):
        img_filepath = os.path.join(image_directory, f"{img_id}.tif")
        if os.path.isfile(img_filepath):
            img = load_img(img_filepath, target_size=img_size)
            img = img_to_array(img) / 255.0  # Normalize
            images.append(img)
            labels_list.append(label)
    return np.array(images), np.array(labels_list)

print("Loading Training Data...")
train_images, train_labels = load_image_data(train_images_path, labels_path, image_dim)

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Data augmentation
data_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = data_gen.flow(X_train, y_train, batch_size=batch_size)
val_generator = data_gen.flow(X_val, y_val, batch_size=batch_size)

# Build the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()

print("Training Model...")
model_history = cnn_model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

# Load test images
def load_test_images(image_directory, img_size):
    image_list = []
    file_names = []
    for filename in os.listdir(image_directory):
        if filename.endswith('.tif'):
            img_filepath = os.path.join(image_directory, filename)
            img = load_img(img_filepath, target_size=img_size)
            img = img_to_array(img) / 255.0  # Normalize
            image_list.append(img)
            file_names.append(filename.split('.')[0])
    return np.array(image_list), file_names

print("Loading Test Data...")
test_images, test_image_ids = load_test_images(test_images_path, image_dim)

print("Generating Predictions...")
test_predictions = cnn_model.predict(test_images)
binary_predictions = (test_predictions > 0.5).astype(int).flatten()

print("Creating Submission File...")
submission_df = pd.DataFrame({
    'id': test_image_ids,
    'label': binary_predictions
})
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
