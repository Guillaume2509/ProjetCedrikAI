import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from PIL import Image


def train_skin_tone_model(folder_path):
    X, y = load_and_preprocess_images(folder_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f »Mean Squared Error: {mean_squared_error(y_test, y_pred)} »)
    return model
    
    
def predict_driver_skin_tone(grayscale_image, skin_tone_model, windshield_shade_model, target_size=(64, 64)):
    # Preprocess the grayscale image
    grayscale_image = grayscale_image.resize(target_size)
    grayscale_array = np.array(grayscale_image).flatten().reshape(1, -1) 

# Example usage
skin_tone_folder = « skin_tone_images »
skin_tone_model = train_skin_tone_model(skin_tone_folder)