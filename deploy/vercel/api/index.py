"""VisionGuard MODEL API on Vercel (Task 7).

Vercel cannot run TensorFlow (a Python function must stay within a few hundred MB), so this API serves
a TFLite copy of the champion model with Google's LiteRT runtime and does the preprocessing with
Pillow + NumPy. Flask (UI) and the database stay on localhost and call this API (Task 8).

    GET  /health          public status
    POST /api/predict     header X-API-Key, multipart field "image" -> JSON verdict
"""
import hmac
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

try:                                         # on Vercel: the small LiteRT runtime
    from ai_edge_litert.interpreter import Interpreter
except ImportError:                          # local check on Windows: the same TFLite kernels via TensorFlow
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
ALLOWED = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "4"))      # Vercel limits request bodies to ~4.5 MB
Image.MAX_IMAGE_PIXELS = 50_000_000


def decode_resize_np(data, size=224):
    """Pillow + NumPy version of src/data.py::decode_and_resize (TensorFlow is not available here).

    Same steps: decode by content, force RGB, scale the longer side to `size` keeping the aspect
    ratio (TF's resize_with_pad arithmetic), bilinear with antialiasing, pad with black.
    """
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        im = im.convert("RGB")
        # TF does this arithmetic in float32 - float64 would round some sizes differently (1-pixel shift)
        w, h, t = np.float32(im.size[0]), np.float32(im.size[1]), np.float32(size)
        ratio = max(w / t, h / t)
        rw_f, rh_f = w / ratio, h / ratio
        rw, rh = max(1, int(np.floor(rw_f))), max(1, int(np.floor(rh_f)))
        pad_w = max(0, int(np.floor((t - rw_f) / np.float32(2))))
        pad_h = max(0, int(np.floor((t - rh_f) / np.float32(2))))
        resized = np.asarray(im.resize((rw, rh), Image.BILINEAR), dtype=np.uint8)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[pad_h:pad_h + rh, pad_w:pad_w + rw] = resized
    return canvas.astype(np.float32)[np.newaxis]               # pixels 0-255; scaling is inside the model


class LiteModel:
    """Loads the .tflite model ONCE per function instance."""

    def __init__(self, model_dir=MODEL_DIR):
        t0 = time.perf_counter()
        self.interpreter = Interpreter(model_path=str(model_dir / "champion.tflite"))
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]["index"]
        self.out = self.interpreter.get_output_details()[0]["index"]
        meta = json.loads((model_dir / "champion.json").read_text())
        self.threshold = float(os.getenv("THRESHOLD") or meta["threshold"])
        self.img_size = int(meta["img_size"])
        self.version = f"{meta['name']} v{meta['version']} ({meta['run_name']}, TFLite)"
        self.lock = threading.Lock()                # one interpreter -> one inference at a time
        self.load_seconds = round(time.perf_counter() - t0, 3)

    def p_ai(self, x):
        with self.lock:
            self.interpreter.set_tensor(self.inp, x)
            self.interpreter.invoke()
            return float(self.interpreter.get_tensor(self.out).ravel()[0])


API_KEY = os.getenv("API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set - refusing to start an unprotected API.")

model = LiteModel()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)


def error(code, message):
    return jsonify(error=message), code


@app.get("/")
@app.get("/health")
def health():
    return jsonify(status="ok", service="VisionGuard model API (L230880_A2_Ops)", runtime="TFLite / LiteRT",
                   model_version=model.version, threshold=model.threshold, img_size=model.img_size,
                   max_upload_mb=MAX_UPLOAD_MB, model_load_seconds=model.load_seconds)


@app.post("/api/predict")
def predict():
    if not hmac.compare_digest(request.headers.get("X-API-Key", ""), API_KEY):   # constant-time comparison
        return error(401, "Invalid or missing API key (send header X-API-Key).")
    file = request.files.get("image")
    if file is None or not file.filename:
        return error(400, "Send the image as multipart form field 'image'.")
    name = secure_filename(file.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED:
        return error(415, f"'.{ext or '?'}' files are not accepted. Allowed: {', '.join(sorted(ALLOWED))}.")
    data = file.read()                                   # memory only, nothing written to disk
    t0 = time.perf_counter()
    try:
        x = decode_resize_np(data, model.img_size)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        return error(422, "The file could not be decoded as an image.")
    p_ai = model.p_ai(x)
    is_ai = p_ai >= model.threshold
    return jsonify(label="AI-generated" if is_ai else "Real",
                   confidence=round(p_ai if is_ai else 1 - p_ai, 4), p_ai=round(p_ai, 4),
                   threshold=model.threshold, model_version=model.version,
                   inference_ms=round((time.perf_counter() - t0) * 1000, 1), filename=name)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return error(413, f"The file is larger than the {MAX_UPLOAD_MB:g} MB limit.")


@app.errorhandler(404)
def not_found(_):
    return error(404, "Unknown endpoint. Use GET /health or POST /api/predict.")
