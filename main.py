import os
import jwt
import datetime
from functools import wraps
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from waitress import serve

# Import ML helpers (your model_utils has the actual logic)
from model.model_utils import (
    ensure_model_downloaded,
    load_vitamin_model,
    load_class_indices,
    load_mapping,
    predict_vitamin,
)

# -------------------------
# Flask + CORS setup
# -------------------------
app = Flask(__name__)

# Use a single origin string (prevents multiple-value header issues).
# Add your exact Netlify domain here and localhost for local testing.
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "https://precious-longma-59eb39.netlify.app")

CORS(
    app,
    resources={r"/*": {"origins": FRONTEND_ORIGIN}},
    supports_credentials=True,
)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "vitamin_detection_secret_key")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# -------------------------
# Database models
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
# ML: lazy global variables
# -------------------------
vitamin_model = None
class_indices = None
vitamin_mapping = None
MODEL_LOADED = False

def ensure_model_loaded():
    """Lazy-load model and mappings when needed. Returns True if loaded."""
    global vitamin_model, class_indices, vitamin_mapping, MODEL_LOADED

    if MODEL_LOADED:
        return True

    # Ensure model file is present (downloads if missing)
    ok = ensure_model_downloaded()
    if not ok:
        print("❌ ensure_model_downloaded failed")
        return False

    # Load model and mapping files (wrap in try/except to avoid crash)
    try:
        vitamin_model = load_vitamin_model()
        # load_class_indices and load_mapping use default paths inside model_utils
        class_indices = load_class_indices()
        vitamin_mapping = load_mapping()
        if vitamin_model is None:
            raise RuntimeError("vitamin_model is None after load")
        MODEL_LOADED = True
        print("✅ Model and mappings loaded into memory.")
        return True
    except Exception as e:
        print("❌ Failed to load model/mappings:", e)
        return False

# -------------------------
# JWT decorator
# -------------------------
def token_required(f):
    from functools import wraps
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
# Auth routes
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

    # If token is bytes (pyjwt older), decode
    if isinstance(token, bytes):
        token = token.decode("utf-8")

    return jsonify({
        "token": token,
        "firstname": user.firstname,
        "lastname": user.lastname,
        "email": user.email
    }), 200

@app.route("/profile", methods=["GET"])
@token_required
def profile(current_user):
    return jsonify({
        "firstname": current_user.firstname,
        "lastname": current_user.lastname,
        "email": current_user.email
    })

# -------------------------
# Prediction endpoint (main)
# -------------------------
@app.route("/detect_vitamin", methods=["POST"])
@token_required
def detect_vitamin(current_user):
    try:
        # lazy load model if not loaded
        if not ensure_model_loaded():
            return jsonify({"message": "Model not available on server"}), 500

        if "image" not in request.files:
            return jsonify({"message": "No image uploaded"}), 400

        image = request.files["image"]
        uploads = "uploads"
        os.makedirs(uploads, exist_ok=True)
        image_path = os.path.join(uploads, image.filename)
        image.save(image_path)

        # perform prediction using model_utils.predict_vitamin
        result = predict_vitamin(vitamin_model, class_indices, vitamin_mapping, image_path)

        # persist user vitamin info
        new_vitamin = UserVitamin(
            user_email=current_user.email,
            vitamin=result.get("mapped_deficiency"),
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        db.session.add(new_vitamin)
        db.session.commit()

        return jsonify(result), 200

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"message": "Server error during prediction", "error": str(e)}), 500

# -------------------------
# /predict alias (some frontends use /predict)
# -------------------------
@app.route("/predict", methods=["POST"])
@token_required
def predict_alias(current_user):
    return detect_vitamin(current_user)

# -------------------------
# User vitamins & delete
# -------------------------
@app.route("/user_vitamins", methods=["GET"])
@token_required
def user_vitamins(current_user):
    records = UserVitamin.query.filter_by(user_email=current_user.email).all()
    return jsonify([{"id": r.id, "vitamin": r.vitamin, "date": r.date} for r in records])

@app.route("/delete_vitamin/<int:id>", methods=["DELETE"])
@token_required
def delete_vitamin(current_user, id):
    record = UserVitamin.query.filter_by(id=id, user_email=current_user.email).first()
    if not record:
        return jsonify({"message": "Vitamin record not found"}), 404
    db.session.delete(record)
    db.session.commit()
    return jsonify({"message": "Deleted successfully"}), 200

# -------------------------
# Health check
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Vitamin Detection Backend Running"})

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting server on port {port}")
    serve(app, host="0.0.0.0", port=port)
