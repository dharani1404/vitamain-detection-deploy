from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import os
import gdown
from waitress import serve
import traceback

# === Import model utilities ===
from model.model_utils import (
    load_vitamin_model,
    load_class_indices,
    load_mapping,
    predict_vitamin_deficiency
)

# === Flask App Config ===
app = Flask(__name__)

# ✅ CORRECT CORS CONFIGURATION
CORS(app, origins=[
    "https://neon-crumble-55544a.netlify.app",
    "http://localhost:3000"
], supports_credentials=True)

SECRET_KEY = os.environ.get("VITAMIN_SECRET_KEY", "vitamin_secret_key")
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "db.sqlite3")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Model Setup ===
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "vitamin_deficiency_model.h5")
JSON_PATH = os.path.join(MODEL_DIR, "class_indices.json")
CSV_PATH = os.path.join(MODEL_DIR, "vitamin_deficiency_data.csv")

model = None
class_indices = None
mapping = None


def get_model():
    """Lazy load model and mappings once."""
    global model, class_indices, mapping

    if model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                print("⬇️ Downloading model from Google Drive...")
                drive_url = "https://drive.google.com/uc?id=1kLvoztjLTDtINxz-Ej_El4Wu1aKL-CUx"
                gdown.download(drive_url, MODEL_PATH, quiet=False, fuzzy=True)
                print("✅ Model downloaded successfully.")

            model = load_vitamin_model(MODEL_PATH)
            class_indices = load_class_indices(JSON_PATH)
            mapping = load_mapping(CSV_PATH)
            print("✅ Model & mappings loaded successfully.")
        except Exception as e:
            print("❌ Error loading model:", e)
            traceback.print_exc()
            raise e

    return model, class_indices, mapping


# === Database Utils ===
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize SQLite database tables."""
    try:
        conn = get_db()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                firstname TEXT,
                lastname TEXT,
                email TEXT UNIQUE,
                password TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_vitamins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                vitamin TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS vitamin_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_vitamin_id INTEGER,
                day_index INTEGER,
                completed INTEGER DEFAULT 0,
                FOREIGN KEY(user_vitamin_id) REFERENCES user_vitamins(id)
            )
        """)

        conn.commit()
        conn.close()
        print("✅ Database initialized successfully.")
    except Exception as e:
        print("❌ Database initialization failed:", e)
        traceback.print_exc()


init_db()


# === ROUTES ===

@app.after_request
def apply_cors_headers(response):
    """Ensure all responses include proper CORS headers."""
    response.headers.add("Access-Control-Allow-Origin", "https://neon-crumble-55544a.netlify.app")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


@app.route("/")
def home():
    return jsonify({"message": "✅ Flask backend is running"}), 200


# --- Register User ---
@app.route("/register", methods=["POST", "OPTIONS"])
def register():
    if request.method == "OPTIONS":
        return '', 200

    try:
        data = request.get_json() or {}
        firstname = data.get("firstname")
        lastname = data.get("lastname")
        email = data.get("email")
        password = data.get("password")

        if not all([firstname, lastname, email, password]):
            return jsonify({"message": "All fields are required"}), 400

        hashed_password = generate_password_hash(password)
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (firstname, lastname, email, password) VALUES (?, ?, ?, ?)",
            (firstname, lastname, email, hashed_password),
        )
        conn.commit()
        conn.close()
        return jsonify({"message": "User registered successfully!"}), 200
    except sqlite3.IntegrityError:
        return jsonify({"message": "User already exists"}), 400
    except Exception as e:
        print("❌ /register error:", e)
        traceback.print_exc()
        return jsonify({"message": "Internal server error"}), 500


# --- Login User ---
@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return '', 200

    try:
        data = request.get_json() or {}
        email = data.get("email")
        password = data.get("password")

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return jsonify({"message": "User not found"}), 400
        if not check_password_hash(row["password"], password):
            return jsonify({"message": "Invalid password"}), 401

        payload = {
            "id": row["id"],
            "email": row["email"],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        if isinstance(token, bytes):
            token = token.decode("utf-8")

        return jsonify({
            "message": "Login successful",
            "token": token,
            "firstname": row["firstname"],
            "lastname": row["lastname"],
            "email": row["email"]
        }), 200
    except Exception as e:
        print("❌ /login error:", e)
        traceback.print_exc()
        return jsonify({"message": "Internal server error"}), 500


# --- Prediction ---
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return '', 200

    try:
        if "image" not in request.files:
            return jsonify({"message": "No image uploaded"}), 400

        img = request.files["image"]
        save_path = os.path.join(UPLOAD_FOLDER, img.filename)
        img.save(save_path)

        model, class_indices, mapping = get_model()
        result = predict_vitamin_deficiency(model, class_indices, mapping, save_path)

        return jsonify({
            "predicted_disease": result["predicted_disease"],
            "vitamin_deficiency": result["mapped_deficiency"],
            "confidence": float(result["confidence"])
        }), 200
    except Exception as e:
        print("❌ /predict error:", e)
        traceback.print_exc()
        return jsonify({"message": f"Prediction error: {str(e)}"}), 500


# === MAIN ENTRY ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    serve(app, host="0.0.0.0", port=port)
