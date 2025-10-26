import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import json
import os
import gdown
import time

# === Paths ===
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "vitamin_deficiency_model.h5")

# === Ensure model directory exists ===
os.makedirs(MODEL_DIR, exist_ok=True)

# === Download model if not present ===
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model from Google Drive...")
    url = "https://drive.google.com/file/d/1kLvoztjLTDtINxz-Ej_El4Wu1aKL-CUx/view?usp=drive_link"
    gdown.download(url, MODEL_PATH, quiet=False)

    # Wait up to 60 seconds for the model to appear
    for _ in range(60):
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            print(f"✅ Model downloaded successfully ({os.path.getsize(MODEL_PATH)/1e6:.2f} MB).")
            break
        time.sleep(1)
    else:
        print("❌ Model download failed or incomplete.")

# === Load class indices (disease label mapping) ===
def load_class_indices(json_path):
    """Load the disease-to-label mapping from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)

# === Load vitamin deficiency mapping ===
def load_mapping(csv_file):
    """Load disease-to-vitamin deficiency mapping from CSV."""
    df = pd.read_csv(csv_file)
    expected_cols = ["Diseases", "Deficiency"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Invalid CSV format. Expected {expected_cols}, got {list(df.columns)}")
    return {row["Diseases"].strip().lower(): row["Deficiency"].strip() for _, row in df.iterrows()}

# === Load trained model ===
def load_vitamin_model(model_path=MODEL_PATH):
    """Safely load the trained TensorFlow model."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        size_mb = os.path.getsize(model_path) / 1e6
        print(f"📂 Loading model from: {model_path} ({size_mb:.2f} MB)")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# === Preprocess uploaded image ===
def preprocess_image(image_path):
    """Read and preprocess image for prediction."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# === Predict disease ===
def predict_disease(model, class_indices, image_path):
    """Predict disease class and confidence score."""
    img = preprocess_image(image_path)
    preds = model.predict(img)
    predicted_index = np.argmax(preds, axis=1)[0]
    class_labels = list(class_indices.keys())
    predicted_class = class_labels[predicted_index]
    confidence = float(np.max(preds))
    return predicted_class, confidence

# === Wrapper for final output ===
def predict_vitamin_deficiency(model, class_indices, mapping, image_path):
    """Predict vitamin deficiency from an image."""
    predicted_disease, confidence = predict_disease(model, class_indices, image_path)
    mapped_deficiency = mapping.get(predicted_disease.lower(), "No mapping found")
    return {
        "predicted_disease": predicted_disease,
        "mapped_deficiency": mapped_deficiency,
        "confidence": confidence
    }
