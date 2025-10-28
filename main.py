import os
import jwt
import datetime
from functools import wraps
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from waitress import serve

# === ML imports ===
from model.model_utils import (
    ensure_model_downloaded,
    load_vitamin_model,
    load_class_indices,
    load_mapping,
    predict_vitamin,
)

# -------------------------
# APP CONFIG
# -------------------------
app = Flask(__name__)

FRONTEND_ORIGIN = os.environ.get(
    "FRONTEND_ORIGIN",
    "https://precious-longma-59eb39.netlify.app"
)

CORS(
    app,
    resources={r"/*": {"origins": [FRONTEND_ORIGIN, "http://localhost:3000"]}},
    supports_credentials=True,
)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# -------------------------
# DATABASE CONFIG
# -------------------------
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "vitamin_detection_secret_key")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# -------------------------
# MODELS
# -------------------------
class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(100))
    lastname = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class UserVitamin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False)
    vitamin = db.Column(db.String(500))
    date = db.Column(db.String(100))

with app.app_context():
    db.create_all()

# -------------------------
# MODEL LOADING
# -------------------------
vitamin_model = None
class_indices = None
vitamin_mapping = None
MODEL_LOADED = False

def ensure_model_loaded():
    global vitamin_model, class_indices, vitamin_mapping, MODEL_LOADED
    if MODEL_LOADED:
        return True
    print("⚙️ Loading vitamin model...")
    ok = ensure_model_downloaded()
    if not ok:
        print("❌ ensure_model_downloaded failed")
        return False
    try:
        vitamin_model = load_vitamin_model()
        class_indices = load_class_indices()
        vitamin_mapping = load_mapping()
        if vitamin_model is None:
            raise RuntimeError("vitamin_model is None after load")
        MODEL_LOADED = True
        print("✅ Model and mappings loaded successfully.")
        return True
    except Exception as e:
        print("❌ Model load error:", e)
        return False

# -------------------------
# JWT DECORATOR
# -------------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            parts = request.headers["Authorization"].split(" ")
            if len(parts) == 2:
                token = parts[1]

        if not token:
            return jsonify({"message": "Missing token"}), 401

        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            current_user = Users.query.filter_by(email=data["email"]).first()
            if not current_user:
                return jsonify({"message": "User not found"}), 404
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401

        return f(current_user, *args, **kwargs)
    return decorated

# -------------------------
# AUTH ROUTES
# -------------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    firstname = data.get("firstname")
    lastname = data.get("lastname")
    email = data.get("email")
    password = data.get("password")

    if not all([firstname, lastname, email, password]):
        return jsonify({"message": "All fields are required"}), 400

    if Users.query.filter_by(email=email).first():
        return jsonify({"message": "Email already registered"}), 400

    hashed_pw = generate_password_hash(password)
    new_user = Users(firstname=firstname, lastname=lastname, email=email, password=hashed_pw)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully!"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")

    user = Users.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"message": "Invalid email or password"}), 401

    token = jwt.encode({
        "email": user.email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }, app.config["SECRET_KEY"], algorithm="HS256")

    if isinstance(token, bytes):
        token = token.decode("utf-8")

    return jsonify({
        "token": token,
        "firstname": user.firstname,
        "lastname": user.lastname,
        "email": user.email
    }), 200

# -------------------------
# PROFILE
# -------------------------
@app.route("/profile", methods=["GET"])
@token_required
def profile(current_user):
    return jsonify({
        "firstname": current_user.firstname,
        "lastname": current_user.lastname,
        "email": current_user.email
    })

# -------------------------
# CORS PREFLIGHT FIX
# -------------------------
@app.route("/detect_vitamin", methods=["OPTIONS"])
def preflight_detect():
    response = jsonify({"message": "CORS preflight OK"})
    response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response, 200

# -------------------------
# DETECTION ROUTE (POST)
# -------------------------
@app.route("/detect_vitamin", methods=["POST"])
@token_required
def detect_vitamin(current_user):
    try:
        if not ensure_model_loaded():
            return jsonify({"message": "Model not available on server"}), 500

        if "image" not in request.files:
            return jsonify({"message": "No image uploaded"}), 400

        image = request.files["image"]
        uploads = "uploads"
        os.makedirs(uploads, exist_ok=True)
        image_path = os.path.join(uploads, image.filename)
        image.save(image_path)

        result = predict_vitamin(vitamin_model, class_indices, vitamin_mapping, image_path)

        new_vitamin = UserVitamin(
            user_email=current_user.email,
            vitamin=result.get("mapped_deficiency"),
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        db.session.add(new_vitamin)
        db.session.commit()

        response = jsonify(result)
        response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response, 200

    except Exception as e:
        print("❌ Prediction error:", e)
        response = jsonify({"message": "Server error during prediction", "error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response, 500

# -------------------------
# HEALTH CHECK
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Vitamin Detection Backend Running"}), 200

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    serve(app, host="0.0.0.0", port=port)
