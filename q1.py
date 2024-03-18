import os
import pandas as pd
import requests
import cv2
import numpy as np
import pickle
from keras.applications import ResNet50

def preprocessing(i_path, t=(224, 224)):#if no value define it will take this value as a default
    try:
        img = cv2.imread(i_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read img: {i_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, t)
        # Convert img to float32 and scale to range [0, 1]
        img = img.astype('float32') / 255.0
        return img
    except Exception as e:
        print(f"Error preprocessing img {i_path}: {e}")
        return None



def extract_img_features(i_folder):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = {}
    for name_image in os.listdir(i_folder):
        i_path = os.path.join(i_folder, name_image)
        image_preprocessed = preprocessing(i_path)
        if image_preprocessed is not None:
            features[name_image] = model.predict(np.expand_dims(image_preprocessed, axis=0)).flatten()
    return features

def normalization(features):
    for name_image, feature_vector in features.items():
        features[name_image] = feature_vector / np.linalg.norm(feature_vector)
    return features

def download(dataset_file, output_folder):
    for index, row in dataset_file.iterrows():
        img_urls = eval(row['Image']) 
        img_id = row['ID']
        for img_url in img_urls:
            i_path = os.path.join(output_folder, f"{img_id}_{img_urls.index(img_url)}.jpg")
            try:
                img = requests.get(img_url).content
                print(f"Downloading this img: {i_path}")
                with open(i_path, 'wb') as f:
                    f.write(img)
                print("downloaded that img")
            except Exception as e:
                print(f"Error downloading img {img_url}: {e}")

csv_file = "dataset.csv"
i_folder = "downloaded_imgs"
output_features_file = "Q1.pkl"  
dataset_file = pd.read_csv(csv_file)

# Download imgs
# print("Downloading imgs..............")
# if not os.path.exists(i_folder):
#         os.makedirs(i_folder)
# download(dataset_file, i_folder)

# Extract img features
print("Extracting features...........")
img_features = extract_img_features(i_folder)

# Normalize img features
print("Normalizing features............")
normalized_img_features = normalization(img_features)

# Save features to pickle file
with open(output_features_file, 'wb') as f:
    pickle.dump(normalized_img_features, f)

print("img features extracted and saved successfully!")