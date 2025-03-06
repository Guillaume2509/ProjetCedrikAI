import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from PIL import Image

def load_and_preprocess_images(folder_path, target_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(« .jpg ») or filename.endswith(« .png »):
            img = Image.open(os.path.join(folder_path, filename)).convert(‘L’)  # Convert to grayscale
            img = img.resize(target_size)  # Resize to target size
            img_array = np.array(img).flatten()  # Flatten to 1D array
            images.append(img_array)
            # Assume skin tone label is in the filename (e.g., « skin_tone_0.75.jpg »)
            label = float(filename.split(« _ »)[-1].split(« . »)[0])
            labels.append(label)
    return np.array(images), np.array(labels)