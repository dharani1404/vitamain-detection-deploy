import os
import time
import json
import cv2
import gdown
import numpy as np
import pandas as pd

MODEL_DIR = os.path.dirname(__file__) or "model"
MODEL_PATH = os.path.join(MODEL_DIR, "vitamin_deficiency_model.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")
CSV_MAPPING_PATH = os.path.join(MODEL_DIR, "vitamin_deficiency_data.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

def ensure_model_downloaded():
    """Download the model from Google Drive if it's missing."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        print("🔽 Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1kLvoztjLTDtINxz-Ej_El4Wu1aKL-CUx"
        try:
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            print("❌ gdown failed:", e)
            return False

        for _ in range(120):
            if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
                print(f"✅ Model downloaded successfully ({os.path.getsize(MODEL_PATH)/1e6:.2f} MB).")
                return True
            time.sleep(1)
        print("❌ Model download failed or incomplete.")
        return False
    else:
        print("✅ Model already present.")
        return True

def load_class_indices(json_path=CLASS_INDICES_PATH):
    with open(json_path, "r") as f:
        return json.load(f)

def load_mapping(csv_file=CSV_MAPPING_PATH):
    df = pd.read_csv(csv_file)
    return {row["Diseases"].strip().lower(): row["Deficiency"].strip() for _, row in df.iterrows()}

def load_vitamin_model():
    try:
        import tensorflow as tf
    except Exception as e:
        print("❌ TensorFlow import failed:", e)
        return None

    try:
        ok = ensure_model_downloaded()
        if not ok:
            print("❌ ensure_model_downloaded returned False")
            return None

        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print(f"📂 Loading model from: {MODEL_PATH} ({size_mb:.2f} MB)")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict_disease(model, class_indices, image_path):
    img = preprocess_image(image_path)
    preds = model.predict(img)
    predicted_index = int(np.argmax(preds, axis=1)[0])
    class_labels = list(class_indices.keys())
    predicted_class = class_labels[predicted_index]
    confidence = float(np.max(preds))
    return predicted_class, confidence

def predict_vitamin_deficiency(model, class_indices, mapping, image_path):
    predicted_disease, confidence = predict_disease(model, class_indices, image_path)
    mapped_deficiency = mapping.get(predicted_disease.lower(), "No mapping found")
    return {
        "predicted_disease": predicted_disease,
        "mapped_deficiency": mapped_deficiency,
        "confidence": confidence
    }

predict_vitamin = predict_vitamin_deficiency
