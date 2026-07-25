import json
import os
import sqlite3
import tempfile
import threading
import webbrowser
from contextlib import closing
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.exceptions import HTTPException

from capture import save_image_from_data_url
from constellation import create_constellation
from database import get_connection, init_db, resolve_db_path
from feature_extraction import extract_features
from hashing import generate_hash
from iris_processing import assess_image_quality, detect_iris


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(PROJECT_ROOT, "database.db")
MATCH_THRESHOLD = 0.62
STRONG_PATTERN_THRESHOLD = 0.78
SIGNATURE_RECOVERY_THRESHOLD = 0.58
SIGNATURE_STRONG_THRESHOLD = 0.72
TOKEN_MAX_AGE_SECONDS = 3600
MIN_PATTERN_LENGTH = 12
GEOMETRY_REJECT_GAP = 500.0
IRIS_CODE_LENGTH = 192
IRIS_SECTORS = 12
SECTOR_TEXTURE_LENGTH = 144
APP_NAME = "IrisPass"
API_SERVICE_NAME = "iris-constellation-password-system"
TOKEN_SALT = "iris-auth"
UPLOAD_DIRECTORY = os.path.join(tempfile.gettempdir(), "iris_project_uploads")


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "iris-constellation-demo-secret")
app.config.update(
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
DATABASE_PATH = resolve_db_path(os.environ.get("DATABASE_PATH", DATABASE_PATH))
database_lock = threading.Lock()
token_serializer = URLSafeTimedSerializer(app.secret_key)


def ensure_database_ready():
    if os.path.exists(DATABASE_PATH):
        return

    with database_lock:
        if not os.path.exists(DATABASE_PATH):
            init_db(DATABASE_PATH)


init_db(DATABASE_PATH)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.before_request
def prepare_database_for_request():
    if request.endpoint != "static":
        ensure_database_ready()


def _split_pattern_components(pattern):
    raw_length = (len(pattern) + 1) // 2
    raw = np.asarray(pattern[:raw_length], dtype=np.float32)
    deltas = np.asarray(pattern[raw_length:], dtype=np.float32)
    return raw, deltas


def _normalized_similarity(reference, candidate):
    if reference.size == 0 or candidate.size == 0:
        return 0.0

    shared_length = min(reference.size, candidate.size)
    reference = reference[:shared_length]
    candidate = candidate[:shared_length]

    mean_distance = float(np.mean(np.abs(reference - candidate)))
    dynamic_range = max(
        float(np.percentile(reference, 95) - np.percentile(reference, 5)),
        float(np.percentile(candidate, 95) - np.percentile(candidate, 5)),
        1.0,
    )
    distance_score = max(0.0, 1.0 - (mean_distance / dynamic_range))

    reference_centered = reference - float(np.mean(reference))
    candidate_centered = candidate - float(np.mean(candidate))
    reference_norm = float(np.linalg.norm(reference_centered))
    candidate_norm = float(np.linalg.norm(candidate_centered))
    if reference_norm < 1e-6 or candidate_norm < 1e-6:
        cosine_score = 0.0
    else:
        cosine = float(np.dot(reference_centered, candidate_centered) / (reference_norm * candidate_norm))
        cosine_score = max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    tolerance = max(dynamic_range * 0.18, 12.0)
    agreement_score = float(np.mean(np.abs(reference - candidate) <= tolerance))

    return (distance_score * 0.35) + (cosine_score * 0.4) + (agreement_score * 0.25)


def _distribution_similarity(reference, candidate):
    if reference.size == 0 or candidate.size == 0:
        return 0.0

    shared_length = min(reference.size, candidate.size)
    reference = np.sort(reference[:shared_length])
    candidate = np.sort(candidate[:shared_length])
    return _normalized_similarity(reference, candidate)


def _capped_distribution_similarity(reference, candidate):
    ordered_score = _normalized_similarity(reference, candidate)
    distribution_score = _distribution_similarity(reference, candidate)
    return max(ordered_score, min(distribution_score * 0.86, ordered_score + 0.08))


def _binary_similarity(reference, candidate):
    if reference.size == 0 or candidate.size == 0:
        return 0.0

    shared_length = min(reference.size, candidate.size)
    reference_bits = reference[:shared_length] >= 500
    candidate_bits = candidate[:shared_length] >= 500
    return float(np.mean(reference_bits == candidate_bits))


def _vector_similarity(reference, candidate):
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.size == 0 or candidate.size == 0:
        return 0.0

    shared_length = min(reference.size, candidate.size)
    if shared_length == 0:
        return 0.0

    reference = reference[:shared_length]
    candidate = candidate[:shared_length]
    mean_gap = float(np.mean(np.abs(reference - candidate)))
    gap_score = max(0.0, 1.0 - (mean_gap / 95.0))

    reference_centered = reference - float(np.mean(reference))
    candidate_centered = candidate - float(np.mean(candidate))
    reference_norm = float(np.linalg.norm(reference_centered))
    candidate_norm = float(np.linalg.norm(candidate_centered))
    if reference_norm < 1e-6 or candidate_norm < 1e-6:
        correlation_score = 0.0
    else:
        correlation = float(np.dot(reference_centered, candidate_centered) / (reference_norm * candidate_norm))
        correlation_score = max(0.0, min(1.0, (correlation + 1.0) / 2.0))

    return (correlation_score * 0.7) + (gap_score * 0.3)


def create_visual_signature(image_path, circles):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None or not circles:
        return []

    height, width = image.shape
    x, y, radius = circles[0]
    x = int(x)
    y = int(y)
    radius = int(radius)

    crop_x1 = max(0, x - int(radius * 2.4))
    crop_y1 = max(0, y - int(radius * 1.7))
    crop_x2 = min(width, x + int(radius * 2.4))
    crop_y2 = min(height, y + int(radius * 1.7))
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return []

    crop = cv2.equalizeHist(crop)
    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    resized = cv2.resize(crop, (32, 24), interpolation=cv2.INTER_AREA)

    mean_value = float(np.mean(resized))
    std_value = float(np.std(resized)) or 1.0
    normalized = np.clip(((resized - mean_value) / std_value) * 38.0 + 128.0, 0, 255)
    return [int(round(value)) for value in normalized.flatten()]


def compare_visual_signatures(reference_signature, candidate_signature):
    if not reference_signature or not candidate_signature:
        return None

    if len(reference_signature) != 768 or len(candidate_signature) != 768:
        return None

    reference = np.asarray(reference_signature, dtype=np.float32).reshape(24, 32)
    candidate = np.asarray(candidate_signature, dtype=np.float32).reshape(24, 32)
    best_score = 0.0

    for y_shift in range(-2, 3):
        for x_shift in range(-3, 4):
            shifted = np.roll(candidate, shift=(y_shift, x_shift), axis=(0, 1))
            pixel_score = _vector_similarity(reference.flatten(), shifted.flatten())
            row_score = _vector_similarity(np.mean(reference, axis=1), np.mean(shifted, axis=1))
            col_score = _vector_similarity(np.mean(reference, axis=0), np.mean(shifted, axis=0))
            hist_reference, _ = np.histogram(reference, bins=16, range=(0, 255), density=True)
            hist_candidate, _ = np.histogram(shifted, bins=16, range=(0, 255), density=True)
            hist_score = _vector_similarity(hist_reference, hist_candidate)
            score = (pixel_score * 0.45) + (row_score * 0.18) + (col_score * 0.18) + (hist_score * 0.19)
            best_score = max(best_score, score)

    return round(best_score, 4)


def _roll_sector_texture(values, sector_shift):
    if values.size < SECTOR_TEXTURE_LENGTH:
        return values

    sector_maps = values[:SECTOR_TEXTURE_LENGTH].reshape(3, 4, IRIS_SECTORS)
    rolled_maps = np.roll(sector_maps, sector_shift, axis=2).reshape(-1)
    return np.concatenate((rolled_maps, values[SECTOR_TEXTURE_LENGTH:]))


def _roll_iris_bits(values, sector_shift):
    if values.size != IRIS_CODE_LENGTH:
        return values

    bit_map = values.reshape(4, IRIS_SECTORS, 4)
    return np.roll(bit_map, sector_shift, axis=1).reshape(-1)


def _best_aligned_scores(reference_texture, candidate_texture, reference_bits, candidate_bits):
    best_texture_score = 0.0
    best_bit_score = 0.0
    best_combined_score = -1.0

    for sector_shift in range(-3, 4):
        shifted_texture = _roll_sector_texture(candidate_texture, sector_shift)
        shifted_bits = _roll_iris_bits(candidate_bits, sector_shift)
        texture_score = _capped_distribution_similarity(reference_texture, shifted_texture)
        bit_score = _binary_similarity(reference_bits, shifted_bits)
        combined_score = (texture_score * 0.55) + (bit_score * 0.45)

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_texture_score = texture_score
            best_bit_score = bit_score

    return best_texture_score, best_bit_score


def compare_patterns(reference_pattern, candidate_pattern):
    if not reference_pattern or not candidate_pattern:
        return 0.0

    reference_raw, reference_deltas = _split_pattern_components(reference_pattern)
    candidate_raw, candidate_deltas = _split_pattern_components(candidate_pattern)

    if reference_raw.size <= 3 or candidate_raw.size <= 3:
        return 0.0

    # The first three raw values describe capture position/scale. They catch extreme capture
    # changes, while the texture and iris-code sections carry the identity decision.
    geometry_reference = reference_raw[:3]
    geometry_candidate = candidate_raw[:3]

    geometry_gap = float(np.mean(np.abs(geometry_reference - geometry_candidate)))
    if geometry_gap > GEOMETRY_REJECT_GAP:
        return 0.0

    geometry_score = max(0.9, 1.0 - (geometry_gap / 1200.0))

    if reference_raw.size > IRIS_CODE_LENGTH + 8 and candidate_raw.size > IRIS_CODE_LENGTH + 8:
        reference_texture = reference_raw[3:-IRIS_CODE_LENGTH]
        candidate_texture = candidate_raw[3:-IRIS_CODE_LENGTH]
        reference_bits = reference_raw[-IRIS_CODE_LENGTH:]
        candidate_bits = candidate_raw[-IRIS_CODE_LENGTH:]
    else:
        reference_texture = reference_raw[3:]
        candidate_texture = candidate_raw[3:]
        reference_bits = np.asarray([], dtype=np.float32)
        candidate_bits = np.asarray([], dtype=np.float32)

    texture_score, bit_score = _best_aligned_scores(
        reference_texture,
        candidate_texture,
        reference_bits,
        candidate_bits,
    )
    delta_score = _capped_distribution_similarity(reference_deltas, candidate_deltas)

    if reference_bits.size and candidate_bits.size:
        if texture_score < 0.35:
            return 0.0

        # Iris-code bits shift heavily between webcam captures, so treat them as
        # supporting evidence instead of a hard reject gate.
        if bit_score >= 0.55:
            biometric_score = (texture_score * 0.47) + (bit_score * 0.43) + (delta_score * 0.1)
        elif bit_score >= 0.40:
            biometric_score = (texture_score * 0.62) + (bit_score * 0.18) + (delta_score * 0.2)
        else:
            biometric_score = (texture_score * 0.72) + (bit_score * 0.08) + (delta_score * 0.2)
    else:
        if texture_score < 0.45 or delta_score < 0.45:
            return 0.0
        biometric_score = (texture_score * 0.72) + (delta_score * 0.28)

    shared_length = min(len(reference_pattern), len(candidate_pattern))
    length_penalty = shared_length / max(len(reference_pattern), len(candidate_pattern))
    combined_score = biometric_score * length_penalty * geometry_score
    # Return the raw score so callers can combine it with visual evidence.
    # Hard rejects above already return 0.0 for geometry/texture/bit failures.
    return round(combined_score, 4)


def compare_biometrics(reference_pattern, candidate_pattern, reference_signature=None, candidate_signature=None):
    pattern_score = compare_patterns(reference_pattern, candidate_pattern)
    signature_score = compare_visual_signatures(reference_signature, candidate_signature)

    if pattern_score >= MATCH_THRESHOLD:
        if signature_score is not None and signature_score >= SIGNATURE_RECOVERY_THRESHOLD:
            combined_score = (pattern_score * 0.65) + (signature_score * 0.35)
            return round(max(pattern_score, combined_score), 4)
        return pattern_score

    if signature_score is not None:
        if pattern_score >= 0.48 and signature_score >= SIGNATURE_STRONG_THRESHOLD:
            combined_score = (pattern_score * 0.45) + (signature_score * 0.55)
            if combined_score >= MATCH_THRESHOLD:
                return round(combined_score, 4)

        if pattern_score >= 0.38 and signature_score >= 0.62:
            combined_score = (pattern_score * 0.4) + (signature_score * 0.6)
            if combined_score >= MATCH_THRESHOLD:
                return round(combined_score, 4)

    return 0.0


def evaluate_biometric_match(reference_pattern, candidate_pattern, reference_signature=None, candidate_signature=None):
    pattern_score = compare_patterns(reference_pattern, candidate_pattern)
    signature_score = compare_visual_signatures(reference_signature, candidate_signature)
    final_score = compare_biometrics(
        reference_pattern,
        candidate_pattern,
        reference_signature,
        candidate_signature,
    )

    return {
        "pattern_score": round(pattern_score, 4),
        "signature_score": signature_score,
        "final_score": round(final_score, 4),
        "accepted": final_score >= MATCH_THRESHOLD,
    }


def get_trimmed_value(value):
    if value is None:
        return ""

    return str(value).strip()


def normalize_image_payload(payload):
    if not isinstance(payload, str):
        return None

    raw_value = payload.strip()
    return raw_value or None


def build_iris_password(image_data):
    image_path = None
    try:
        image_path = save_image_from_data_url(
            normalize_image_payload(image_data),
            output_dir=UPLOAD_DIRECTORY,
        )
        if not image_path:
            return None, "Camera image was not received. Capture a photo and try again."

        quality = assess_image_quality(image_path)
        if not quality["ok"]:
            return None, quality["reason"]

        circles = detect_iris(image_path)
        if not circles:
            return None, "Iris not detected. Capture a clearer eye image and try again."

        features = extract_features(image_path, circles)
        if not features:
            return None, "No iris features were extracted from the captured image."

        constellation_pattern = create_constellation(features)
        if len(constellation_pattern) < MIN_PATTERN_LENGTH:
            return None, "Captured image is too weak for biometric verification. Keep one eye open and try again."

        visual_signature = create_visual_signature(image_path, circles)
        iris_hash = generate_hash(constellation_pattern)

        return {
            "image_path": image_path,
            "quality": quality,
            "circles": circles,
            "features": features,
            "pattern": constellation_pattern,
            "signature": visual_signature,
            "hash": iris_hash,
        }, None
    except Exception:
        app.logger.exception("Iris capture processing failed")
        return None, "Iris verification could not process this capture. Capture a clearer eye image and try again."
    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                app.logger.exception("Temporary iris capture cleanup failed")


def _is_missing_schema_error(error):
    return "no such table" in str(error).lower()


def fetch_user_record(username):
    ensure_database_ready()
    try:
        with closing(get_connection(DATABASE_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT hash, pattern, signature FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
    except sqlite3.OperationalError as error:
        if not _is_missing_schema_error(error):
            raise

        init_db(DATABASE_PATH)
        with closing(get_connection(DATABASE_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT hash, pattern, signature FROM users WHERE username = ?", (username,))
            row = cur.fetchone()

    return row if row else None


def register_user(username, iris_hash, iris_pattern, iris_signature=None):
    ensure_database_ready()
    try:
        with closing(get_connection(DATABASE_PATH)) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO users (username, hash, pattern, signature) VALUES (?, ?, ?, ?)",
                (username, iris_hash, json.dumps(iris_pattern), json.dumps(iris_signature or [])),
            )
            conn.commit()
    except sqlite3.OperationalError as error:
        if not _is_missing_schema_error(error):
            raise

        init_db(DATABASE_PATH)
        with closing(get_connection(DATABASE_PATH)) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO users (username, hash, pattern, signature) VALUES (?, ?, ?, ?)",
                (username, iris_hash, json.dumps(iris_pattern), json.dumps(iris_signature or [])),
            )
            conn.commit()


def parse_stored_pattern(serialized_pattern):
    if not serialized_pattern:
        return None

    try:
        pattern = json.loads(serialized_pattern)
    except json.JSONDecodeError:
        return None

    return pattern if isinstance(pattern, list) else None


def parse_stored_signature(serialized_signature):
    if not serialized_signature:
        return None

    try:
        signature = json.loads(serialized_signature)
    except json.JSONDecodeError:
        return None

    return signature if isinstance(signature, list) else None


def build_api_response(status, message, **extra):
    body = {"status": status, "message": message}
    body.update(extra)
    return jsonify(body)


def is_api_request():
    return request.path.startswith("/api/")


def build_page_error_response(message, status_code=500):
    if request.path.startswith("/login"):
        return render_template("login.html", error=message), status_code

    if request.path.startswith("/register") or request.path == "/":
        return render_template("register.html", error=message), status_code

    return render_template("error.html", message=message), status_code


@app.errorhandler(HTTPException)
def handle_http_error(error):
    message = error.description or "The requested page could not be completed."
    status_code = error.code or 500

    if is_api_request():
        return build_api_response("fail", message), status_code

    return build_page_error_response(message, status_code)


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    app.logger.exception("Unhandled application error")
    message = "Something went wrong. Please try again."

    if is_api_request():
        return build_api_response("fail", message), 500

    return build_page_error_response(message)


def issue_auth_token(username):
    return token_serializer.dumps({"username": username}, salt=TOKEN_SALT)


def read_auth_token(token):
    try:
        payload = token_serializer.loads(
            token,
            salt=TOKEN_SALT,
            max_age=TOKEN_MAX_AGE_SECONDS,
        )
    except SignatureExpired:
        return None, "Token expired"
    except BadSignature:
        return None, "Invalid token"

    return payload, None


def current_timestamp():
    return datetime.now().strftime("%d %b %Y, %I:%M %p")


def request_json_payload():
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def build_session_activity(username):
    return [
        {
            "label": "Biometric match accepted",
            "detail": f"{username} was verified against the enrolled iris pattern.",
            "time": session.get("login_time", "Just now"),
        },
        {
            "label": "Protected session issued",
            "detail": "Access was granted to the authenticated dashboard workspace.",
            "time": session.get("login_time", "Just now"),
        },
        {
            "label": "Integration-ready token flow",
            "detail": "Signed session verification is available for connected applications.",
            "time": "Active",
        },
    ]


@app.route("/")
def home():
    return render_template("register.html")


@app.route("/integration-demo")
def integration_demo():
    return render_template("integration_demo.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    return build_api_response(
        "success",
        "Iris API is running",
        service=API_SERVICE_NAME,
        app_name=APP_NAME,
    )


@app.route("/api/meta", methods=["GET"])
def api_meta():
    return build_api_response(
        "success",
        "Integration metadata",
        service=API_SERVICE_NAME,
        app_name=APP_NAME,
        version="1.0",
        endpoints={
            "health": "/api/health",
            "register": "/api/register",
            "login": "/api/login",
            "verify_session": "/api/session/verify",
        },
        payload_examples={
            "register_or_login": {
                "username": "vipin",
                "image_data": "data:image/jpeg;base64,...",
            },
            "verify_session": {
                "auth_token": "signed-token-from-login",
            },
        },
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = get_trimmed_value(request.form.get("username"))
        image_data = request.form.get("image_data")

        if not username:
            return render_template(
                "register.html",
                error="Please enter a username before capturing the iris sample.",
            )

        if fetch_user_record(username):
            return render_template(
                "register.html",
                error="This username is already registered. Choose a different username or log in instead.",
                username=username,
            )

        iris_data, error = build_iris_password(image_data)
        if error:
            return render_template("register.html", error=error, username=username)

        register_user(username, iris_data["hash"], iris_data["pattern"], iris_data.get("signature"))
        return render_template("success.html", username=username)

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = get_trimmed_value(request.form.get("username"))
        image_data = request.form.get("image_data")

        if not username:
            return render_template("login.html", error="Please enter your username.")

        user_record = fetch_user_record(username)
        if not user_record:
            return render_template(
                "login.html",
                error="This username is not registered yet. Please register first.",
                username=username,
            )

        _, stored_pattern_raw, stored_signature_raw = user_record
        stored_pattern = parse_stored_pattern(stored_pattern_raw)
        if not stored_pattern:
            return render_template(
                "login.html",
                error="This account was saved with an older biometric format. Please register again.",
                username=username,
            )
        stored_signature = parse_stored_signature(stored_signature_raw)

        iris_data, error = build_iris_password(image_data)
        if error:
            return render_template("login.html", error=error, username=username)

        match = evaluate_biometric_match(
            stored_pattern,
            iris_data["pattern"],
            stored_signature,
            iris_data.get("signature"),
        )
        if match["accepted"]:
            session["username"] = username
            session["match_score"] = round(match["final_score"], 2)
            session["login_time"] = current_timestamp()
            return redirect(url_for("login_success", username=username))

        return render_template(
            "login_fail.html",
            username=username,
            score=round(match["final_score"], 2),
            pattern_score=match["pattern_score"],
            signature_score=match["signature_score"],
        )

    return render_template("login.html")


@app.route("/login_success/<username>")
def login_success(username):
    score = session.get("match_score")
    return render_template("login_success.html", username=username, score=score)


@app.route("/dashboard")
def dashboard():
    username = session.get("username")
    if not username:
        return redirect(url_for("login"))
    score = session.get("match_score")
    return render_template(
        "dashboard.html",
        username=username,
        score=score,
        login_time=session.get("login_time", "Just now"),
        activity=build_session_activity(username),
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/api/register", methods=["POST"], provide_automatic_options=False)
def api_register():
    payload = request_json_payload()
    username = get_trimmed_value(payload.get("username"))
    image_data = payload.get("image_data")

    if not username:
        return build_api_response("fail", "Username is required"), 400

    if fetch_user_record(username):
        return build_api_response(
            "fail",
            "Username already registered",
            username=username,
        ), 409

    iris_data, error = build_iris_password(image_data)
    if error:
        return build_api_response("fail", error), 400

    register_user(username, iris_data["hash"], iris_data["pattern"], iris_data.get("signature"))
    return build_api_response("success", "User registered", username=username)


@app.route("/api/register", methods=["OPTIONS"])
def api_register_options():
    return ("", 204)


@app.route("/api/login", methods=["POST"], provide_automatic_options=False)
def api_login():
    payload = request_json_payload()
    username = get_trimmed_value(payload.get("username"))
    image_data = payload.get("image_data")

    if not username:
        return build_api_response("fail", "Username is required"), 400

    user_record = fetch_user_record(username)
    if not user_record:
        return build_api_response("fail", "User not found"), 404

    _, stored_pattern_raw, stored_signature_raw = user_record
    stored_pattern = parse_stored_pattern(stored_pattern_raw)
    if not stored_pattern:
        return build_api_response(
            "fail",
            "User must register again with the latest biometric format",
        ), 409
    stored_signature = parse_stored_signature(stored_signature_raw)

    iris_data, error = build_iris_password(image_data)
    if error:
        return build_api_response("fail", error), 400

    match = evaluate_biometric_match(
        stored_pattern,
        iris_data["pattern"],
        stored_signature,
        iris_data.get("signature"),
    )
    if match["accepted"]:
        return build_api_response(
            "success",
            "Authentication successful",
            username=username,
            score=round(match["final_score"], 2),
            pattern_score=match["pattern_score"],
            signature_score=match["signature_score"],
            auth_token=issue_auth_token(username),
            expires_in=TOKEN_MAX_AGE_SECONDS,
        )

    return build_api_response(
        "fail",
        "Authentication failed",
        username=username,
        score=round(match["final_score"], 2),
        pattern_score=match["pattern_score"],
        signature_score=match["signature_score"],
    ), 401


@app.route("/api/login", methods=["OPTIONS"])
def api_login_options():
    return ("", 204)


@app.route("/api/session/verify", methods=["POST"], provide_automatic_options=False)
def api_session_verify():
    payload = request_json_payload()
    token = get_trimmed_value(payload.get("auth_token"))

    if not token:
        return build_api_response("fail", "auth_token is required"), 400

    token_data, error = read_auth_token(token)
    if error:
        return build_api_response("fail", error), 401

    username = token_data.get("username")
    return build_api_response(
        "success",
        "Token is valid",
        username=username,
        expires_in=TOKEN_MAX_AGE_SECONDS,
    )


@app.route("/api/session/verify", methods=["OPTIONS"])
def api_session_verify_options():
    return ("", 204)


def open_local_browser(port):
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1, lambda: webbrowser.open(url)).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        open_local_browser(port)
    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )
