import os
import time
import json
import cv2
import gdown
import numpy as np
import pandas as pd
import tensorflow as tf

# === Paths ===
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "vitamin_deficiency_model.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")
CSV_MAPPING_PATH = os.path.join(MODEL_DIR, "vitamin_deficiency_data.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# === Ensure model availability ===
def ensure_model_downloaded():
    """Download the model from Google Drive if it's missing."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        print("🔽 Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1kLvoztjLTDtINxz-Ej_El4Wu1aKL-CUx"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

        # Wait until file is available
        for _ in range(60):
            if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
                print(f"✅ Model downloaded successfully ({os.path.getsize(MODEL_PATH)/1e6:.2f} MB).")
                return True
            time.sleep(1)
        print("❌ Model download failed or incomplete.")
        return False
    else:
        print("✅ Model already present.")
        return True


# === Loaders ===
def load_class_indices(json_path=CLASS_INDICES_PATH):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Class index file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def load_mapping(csv_file=CSV_MAPPING_PATH):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Mapping CSV not found: {csv_file}")
    df = pd.read_csv(csv_file)
    expected_cols = ["Diseases", "Deficiency"]
    if list(df.columns) != expected_cols:
        raise ValueError(f"Invalid CSV format. Expected {expected_cols}, got {list(df.columns)}")
    return {row["Diseases"].strip().lower(): row["Deficiency"].strip() for _, row in df.iterrows()}


# === Load model ===
def load_vitamin_model():
    """Load TensorFlow model safely."""
    try:
        ensure_model_downloaded()
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print(f"📂 Loading model from: {MODEL_PATH} ({size_mb:.2f} MB)")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


# === Preprocessing ===
def preprocess_image(image_path):
    """Preprocess image before feeding to model."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)


# === Prediction ===
def predict_disease(model, class_indices, image_path):
    """Predict disease and confidence."""
    if model is None:
        raise ValueError("Model not loaded.")
    img = preprocess_image(image_path)
    preds = model.predict(img)
    predicted_index = np.argmax(preds, axis=1)[0]
    class_labels = list(class_indices.keys())
    predicted_class = class_labels[predicted_index]
    confidence = float(np.max(preds))
    return predicted_class, confidence


def predict_vitamin_deficiency(model, class_indices, mapping, image_path):
    """Predict vitamin deficiency from an image."""
    predicted_disease, confidence = predict_disease(model, class_indices, image_path)
    mapped_deficiency = mapping.get(predicted_disease.lower(), "No mapping found")
    return {
        "predicted_disease": predicted_disease,
        "mapped_deficiency": mapped_deficiency,
        "confidence": confidence
    }


# === Exported alias for main.py ===
predict_vitamin = predict_vitamin_deficiency
