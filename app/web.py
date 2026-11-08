"""Task 4.2 / 5 - VisionGuard Flask app: upload, result, history, health, JSON API."""
import base64
import hmac
import io
import logging

from flask import Flask, Request, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from app.config import Config
from app.db import Database
from app.predictor import LocalPredictor, RemotePredictor

TF_FORMATS = {"JPEG", "PNG", "GIF", "BMP"}        # formats tf.io.decode_image reads directly
Image.MAX_IMAGE_PIXELS = 50_000_000               # refuse "decompression bomb" images
log = logging.getLogger("visionguard")


class UploadError(Exception):
    def __init__(self, status, title, message):
        super().__init__(message)
        self.status, self.title, self.message = status, title, message


class InMemoryRequest(Request):
    """Keep uploaded files in RAM (Werkzeug normally spools files > 500 KB to a temp file)."""
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


def check_upload(file, allowed):
    """Validate one uploaded file. Returns (safe_name, image_bytes, format, width, height, size)."""
    if file is None or not file.filename:
        raise UploadError(400, "No file selected", "Please choose an image file to upload.")
    name = secure_filename(file.filename) or "upload"          # strips paths like ../../ and odd characters
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in allowed:
        raise UploadError(415, "Unsupported file type",
                          f"'.{ext or '?'}' files are not accepted. Please upload: {', '.join(sorted(allowed))}.")
    data = file.read()                                          # kept in memory only - never written to disk
    if not data:
        raise UploadError(400, "Empty file", "The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()                                         # structure check
        with Image.open(io.BytesIO(data)) as im:
            im.load()                                           # real decode of the pixels
            fmt, (w, h) = im.format, im.size
            if fmt not in TF_FORMATS:                           # e.g. WEBP: lossless re-encode for TensorFlow
                buf = io.BytesIO()
                im.convert("RGB").save(buf, "PNG")
                data_for_model = buf.getvalue()
            else:
                data_for_model = data
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise UploadError(422, "Not a valid image",
                          "The file has an image extension, but its contents could not be decoded "
                          "as an image. It may be corrupted or not really an image.")
    return name, data_for_model, fmt, w, h, len(data)


def thumbnail_data_uri(data):
    """Small preview embedded in the result page (memory only, nothing saved)."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((480, 480))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def create_app(config=Config):
    app = Flask(__name__)
    app.request_class = InMemoryRequest                    # uploads never touch the disk
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = int(config.MAX_UPLOAD_MB * 1024 * 1024)   # -> 413 if larger

    db = Database(config.DATABASE_URL)
    if config.INFERENCE_MODE == "remote":
        predictor = RemotePredictor(config.MODEL_API_URL, config.API_KEY)
    else:
        predictor = LocalPredictor(config.MODEL_PATH, float(config.THRESHOLD) if config.THRESHOLD else None)
    log.info("VisionGuard ready: inference=%s, database=%s", config.INFERENCE_MODE,
             config.DATABASE_URL.split("@")[-1])                 # never log the DB password

    def run_prediction(file, source):
        name, data, fmt, w, h, size = check_upload(file, config.ALLOWED_EXTENSIONS)
        try:
            result = predictor.predict(data, name)
        except ValueError as e:
            raise UploadError(422, "Not a valid image", str(e))
        except RuntimeError as e:
            raise UploadError(502, "Model service unavailable", str(e))
        row_id = db.save(filename=name, file_size=size, width=w, height=h, image_format=fmt,
                         label=result["label"], confidence=result["confidence"], p_ai=result["p_ai"],
                         threshold=result["threshold"], model_version=result["model_version"],
                         inference_ms=result.get("inference_ms"), source=source)
        log.info("prediction id=%s file=%s label=%s confidence=%.3f source=%s",
                 row_id, name, result["label"], result["confidence"], source)
        return {"id": row_id, "filename": name, "width": w, "height": h, "image_format": fmt,
                "file_size": size, **{k: result[k] for k in ("label", "confidence", "p_ai", "threshold",
                                                             "model_version", "inference_ms") if k in result}}, data

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=config.MAX_UPLOAD_MB,
                               allowed=sorted(config.ALLOWED_EXTENSIONS))

    @app.post("/predict")
    def predict():
        result, data = run_prediction(request.files.get("image"), "web")
        return render_template("result.html", r=result, preview=thumbnail_data_uri(data))

    @app.get("/history")
    def history():
        return render_template("history.html", rows=db.recent(100), total=db.count())

    @app.get("/health")
    def health():
        body, code = {"status": "ok", "inference": predictor.health(), "database": "ok"}, 200
        try:
            db.ping()
        except Exception as e:                               # noqa: BLE001 - report, don't crash
            body.update(status="degraded", database=f"error: {type(e).__name__}")
            code = 503
        return jsonify(body), code

    @app.post("/api/predict")
    def api_predict():
        key = request.headers.get("X-API-Key", "")
        if not config.API_KEY or not hmac.compare_digest(key, config.API_KEY):
            return jsonify(error="Invalid or missing API key (send header X-API-Key)."), 401
        result, _ = run_prediction(request.files.get("image"), "api")
        return jsonify(result)

    # ---------- friendly error pages (JSON for /api/*) ----------
    def error_response(code, title, message):
        if request.path.startswith("/api/"):
            return jsonify(error=message), code
        return render_template("error.html", code=code, title=title, message=message), code

    @app.errorhandler(UploadError)
    def handle_upload_error(e):
        log.warning("rejected upload: %s (%s)", e.title, e.status)
        return error_response(e.status, e.title, e.message)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(e):
        log.warning("rejected upload: too large")
        return error_response(413, "File too large",
                              f"The file is larger than the {config.MAX_UPLOAD_MB:g} MB limit. "
                              "Please upload a smaller image.")

    @app.errorhandler(HTTPException)
    def handle_http(e):
        return error_response(e.code, e.name, e.description)

    @app.errorhandler(Exception)
    def handle_unexpected(e):
        log.exception("unexpected error")
        return error_response(500, "Something went wrong",
                              "An unexpected error occurred. Please try again later.")

    return app
