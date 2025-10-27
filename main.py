import os
import jwt
import datetime
from functools import wraps
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from waitress import serve
from model.model_utils import process_vitamin_image
  # ✅ your existing model file

# --------------------------------------------------
# ✅ App + Database setup
# --------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

app.config["SECRET_KEY"] = "vitamin_detection_secret_key"  # keep this same everywhere
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# --------------------------------------------------
# ✅ Database Models
# --------------------------------------------------
class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(100))
    lastname = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class UserVitamin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False)
    vitamin = db.Column(db.String(100))
    date = db.Column(db.String(100))


with app.app_context():
    db.create_all()


# --------------------------------------------------
# ✅ JWT Decorator
# --------------------------------------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]
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


# --------------------------------------------------
# ✅ Register
# --------------------------------------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
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


# --------------------------------------------------
# ✅ Login
# --------------------------------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = Users.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"message": "Invalid email or password"}), 401

    token = jwt.encode(
        {
            "email": user.email,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
        },
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )

    return jsonify({
        "token": token,
        "firstname": user.firstname,
        "lastname": user.lastname,
        "email": user.email
    }), 200


# --------------------------------------------------
# ✅ Profile
# --------------------------------------------------
@app.route("/profile", methods=["GET"])
@token_required
def profile(current_user):
    return jsonify({
        "firstname": current_user.firstname,
        "lastname": current_user.lastname,
        "email": current_user.email
    })


# --------------------------------------------------
# ✅ Vitamin Detection
# --------------------------------------------------
@app.route("/detect_vitamin", methods=["POST"])
@token_required
def detect_vitamin(current_user):
    if "image" not in request.files:
        return jsonify({"message": "No image uploaded"}), 400

    image = request.files["image"]
    vitamin = process_vitamin_image(image)

    new_vitamin = UserVitamin(
        user_email=current_user.email,
        vitamin=vitamin,
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    db.session.add(new_vitamin)
    db.session.commit()

    return jsonify({"vitamin": vitamin}), 200


# --------------------------------------------------
# ✅ Fetch user’s vitamins
# --------------------------------------------------
@app.route("/user_vitamins", methods=["GET"])
@token_required
def user_vitamins(current_user):
    records = UserVitamin.query.filter_by(user_email=current_user.email).all()
    return jsonify([
        {"id": r.id, "vitamin": r.vitamin, "date": r.date}
        for r in records
    ])


# --------------------------------------------------
# ✅ Delete vitamin record
# --------------------------------------------------
@app.route("/delete_vitamin/<int:id>", methods=["DELETE"])
@token_required
def delete_vitamin(current_user, id):
    record = UserVitamin.query.filter_by(id=id, user_email=current_user.email).first()
    if not record:
        return jsonify({"message": "Vitamin record not found"}), 404

    db.session.delete(record)
    db.session.commit()
    return jsonify({"message": "Deleted successfully"}), 200


# --------------------------------------------------
# ✅ Health check
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Vitamin Detection Backend Running"})


# --------------------------------------------------
# ✅ Run on Render or local
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
