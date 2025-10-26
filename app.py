from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import os
import gdown  # 👈 for Google Drive download

# === Import model utilities ===
from model.model_utils import (
    load_vitamin_model,
    load_class_indices,
    load_mapping,
    predict_vitamin_deficiency
)

# === Configuration ===
app = Flask(__name__)
CORS(app)
SECRET_KEY = os.environ.get("VITAMIN_SECRET_KEY", "vitamin_secret_key")
DB_PATH = os.path.join(os.path.dirname(__file__), "db.sqlite3")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Google Drive Model Download ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "vitamin_deficiency_model.h5")

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    drive_url = "https://drive.google.com/uc?id=1kLvoztjLTDtINxz-Ej_El4Wu1aKL-CUx"  # 👈 direct file ID version
    gdown.download(drive_url, MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully.")

# === Paths for class indices and mapping ===
JSON_PATH = os.path.join(MODEL_DIR, "class_indices.json")
CSV_PATH = os.path.join(MODEL_DIR, "vitamin_deficiency_data.csv")

# === Load model and metadata ===
model = load_vitamin_model(MODEL_PATH)
class_indices = load_class_indices(JSON_PATH)
mapping = load_mapping(CSV_PATH)

# === Database Utility ===
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# === Initialize DB ===
def init_db():
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

init_db()

# === Helper to decode JWT ===
def decode_token(request):
    token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    elif auth_header:
        token = auth_header
    if not token:
        return None, jsonify({"message": "No token provided"}), 403
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded, None, None
    except jwt.ExpiredSignatureError:
        return None, jsonify({"message": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return None, jsonify({"message": "Invalid token"}), 401

# === Routes ===
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    firstname = data.get("firstname")
    lastname = data.get("lastname")
    email = data.get("email")
    password = data.get("password")

    if not firstname or not lastname or not email or not password:
        return jsonify({"message": "All fields are required"}), 400

    hashed_password = generate_password_hash(password)
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (firstname, lastname, email, password) VALUES (?, ?, ?, ?)",
            (firstname, lastname, email, hashed_password),
        )
        conn.commit()
        conn.close()
        return jsonify({"message": "User registered successfully!"})
    except sqlite3.IntegrityError:
        return jsonify({"message": "User already exists"}), 400

@app.route("/login", methods=["POST"])
def login():
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
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"message": "No image uploaded"}), 400
    img = request.files["image"]
    save_path = os.path.join(UPLOAD_FOLDER, img.filename)
    img.save(save_path)

    try:
        result = predict_vitamin_deficiency(model, class_indices, mapping, save_path)
        return jsonify({
            "predicted_disease": result["predicted_disease"],
            "vitamin_deficiency": result["mapped_deficiency"],
            "confidence": float(result["confidence"])
        })
    except Exception as e:
        return jsonify({"message": f"Prediction error: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask backend for Vitamin Detection running successfully!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
