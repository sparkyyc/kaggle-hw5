# Import necessary libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def batch_image_loader(image_directory, labels_path, image_size, batch_size):
    labels_df = pd.read_csv(labels_path)
    ids = labels_df['id'].tolist()
    labels = labels_df['label'].tolist()
    
    total_images = len(ids)
    image_list = []
    label_list = []

    for start_idx in range(0, total_images, batch_size):
        batch_ids = ids[start_idx:start_idx + batch_size]
        batch_labels = labels[start_idx:start_idx + batch_size]
        
        for img_id, label in zip(batch_ids, batch_labels):
            img_filepath = os.path.join(image_directory, f"{img_id}.tif")
            if os.path.isfile(img_filepath):
                img = load_img(img_filepath, target_size=image_size)
                img = img_to_array(img) / 255.0  # Normalization
                image_list.append(img)
                label_list.append(label)
                
        yield np.array(image_list), np.array(label_list)
        image_list = []
        label_list = []

def load_test_subset(image_directory, labels_path, image_size, subset_count):
    labels_df = pd.read_csv(labels_path).head(subset_count)
    ids = labels_df['id'].tolist()
    labels = labels_df['label'].tolist()
    
    image_list = []
    label_list = []

    for img_id, label in zip(ids, labels):
        img_filepath = os.path.join(image_directory, f"{img_id}.tif")
        if os.path.isfile(img_filepath):
            img = load_img(img_filepath, target_size=image_size)
            img = img_to_array(img) / 255.0  # Normalization
            image_list.append(img)
            label_list.append(label)

    return np.array(image_list), np.array(label_list)

train_images_path = 'histopathologic-cancer-detection/train'
labels_path = 'histopathologic-cancer-detection/train_labels.csv'
image_dim = (64, 64)  # Reduced image dimensions for testing

print("Loading Training Data...")
batch_size = 1000
all_train_images = []
all_train_labels = []

for img_batch, label_batch in batch_image_loader(train_images_path, labels_path, image_dim, batch_size):
    all_train_images.extend(img_batch)
    all_train_labels.extend(label_batch)

all_train_images = np.array(all_train_images)
all_train_labels = np.array(all_train_labels)

X_train, X_val, y_train, y_val = train_test_split(all_train_images, all_train_labels, test_size=0.2, random_state=42)

data_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

data_gen.fit(X_train)

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()

print("Training Model...")
model_history = cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

def load_test_images(image_directory, image_size):
    image_list = []
    for filename in os.listdir(image_directory):
        if filename.endswith('.tif'):
            img_filepath = os.path.join(image_directory, filename)
            img = load_img(img_filepath, target_size=image_size)
            img = img_to_array(img) / 255.0  # Normalization
            image_list.append(img)
    return np.array(image_list)

test_images_path = 'histopathologic-cancer-detection/test'
print("Loading Test Data...")
test_images = load_test_images(test_images_path, image_dim)

print("Generating Predictions...")
test_predictions = cnn_model.predict(test_images)
binary_predictions = (test_predictions > 0.5).astype(int)

print("Creating Submission File...")
sample_sub_path = 'histopathologic-cancer-detection/sample_submission.csv'
submission_df = pd.read_csv(sample_sub_path)
submission_df['label'] = binary_predictions
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
