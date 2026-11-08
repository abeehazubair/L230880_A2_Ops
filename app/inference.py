"""Task 4.1 - VisionGuard inference module.

Loads the champion Keras model ONCE when the app starts, then classifies raw image bytes
using exactly the training preprocessing (src/data.py: decode_and_resize).
    python -m app.inference <image> [<image> ...]      (quick console test)
"""
import json
import os
import sys
import time
from pathlib import Path

import tensorflow as tf

from src.data import decode_and_resize          # SAME function as in training - no copy

DEFAULT_MODEL = "models/champion"


class VisionGuardModel:
    def __init__(self, model_path=DEFAULT_MODEL, threshold=None):
        t0 = time.perf_counter()
        self.model = tf.keras.models.load_model(model_path, compile=False)
        meta_path = Path(model_path).with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        # an explicit threshold (e.g. from an environment variable) wins over the registry value
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        self.img_size = int(meta.get("img_size", self.model.input_shape[1]))
        self.version = f"{meta.get('name', 'VisionGuard')} v{meta.get('version', '?')} ({meta.get('run_name', 'local file')})"
        self.model(tf.zeros((1, self.img_size, self.img_size, 3)), training=False)   # warm-up
        self.load_seconds = time.perf_counter() - t0

    def predict_bytes(self, data: bytes) -> dict:
        """Raw file bytes -> verdict. Raises ValueError if the bytes are not a decodable image."""
        t0 = time.perf_counter()
        try:
            img = decode_and_resize(tf.constant(data), self.img_size)
        except (tf.errors.InvalidArgumentError, tf.errors.UnknownError) as e:
            raise ValueError("The file could not be decoded as an image.") from e
        p_ai = float(self.model(tf.cast(img, tf.float32)[tf.newaxis], training=False).numpy().ravel()[0])
        is_ai = p_ai >= self.threshold
        return {"label": "AI-generated" if is_ai else "Real",
                "confidence": round(p_ai if is_ai else 1 - p_ai, 4),
                "p_ai": round(p_ai, 4),
                "threshold": self.threshold,
                "model_version": self.version,
                "inference_ms": round((time.perf_counter() - t0) * 1000, 1)}


_model = None


def get_model():
    """The one shared model instance for the whole app (created on first call)."""
    global _model
    if _model is None:
        threshold = os.getenv("THRESHOLD")
        _model = VisionGuardModel(os.getenv("MODEL_PATH", DEFAULT_MODEL),
                                  float(threshold) if threshold else None)
    return _model


if __name__ == "__main__":
    m = get_model()
    print(f"Loaded {m.version} in {m.load_seconds:.1f}s  (threshold {m.threshold}, size {m.img_size})")
    for path in sys.argv[1:]:
        r = m.predict_bytes(Path(path).read_bytes())
        print(f"{Path(path).name:45s} -> {r['label']:12s} confidence {r['confidence']:.1%}  "
              f"P(AI)={r['p_ai']:.4f}  ({r['inference_ms']} ms)")
