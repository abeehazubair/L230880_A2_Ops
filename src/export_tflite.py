"""Task 7 - convert the champion model to TFLite for the Vercel model API.

Reads models/champion (SavedModel downloaded by src/fetch_model.py) and writes
models/champion.tflite (float32, no quantisation -> same accuracy). The preprocessing layer
(x/127.5 - 1) is part of the model, so it is inside the converted graph too.
The .tflite file and champion.json are copied into deploy/vercel/model/ for deployment.
    python src/export_tflite.py
"""
import shutil
from pathlib import Path

import tensorflow as tf

SRC = Path("models/champion")
OUT = Path("models/champion.tflite")
DEPLOY = Path("deploy/vercel/model")

model = tf.keras.models.load_model(SRC, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)     # float32: no optimizations
OUT.write_bytes(converter.convert())
print(f"Keras model : {sum(f.stat().st_size for f in SRC.rglob('*') if f.is_file()) / 1e6:.1f} MB (SavedModel)")
print(f"TFLite model: {OUT.stat().st_size / 1e6:.1f} MB -> {OUT.as_posix()}")

interpreter = tf.lite.Interpreter(model_path=str(OUT))
interpreter.allocate_tensors()
i, o = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
print(f"input  {i['shape']} {i['dtype'].__name__}   output {o['shape']} {o['dtype'].__name__}")

DEPLOY.mkdir(parents=True, exist_ok=True)
shutil.copy(OUT, DEPLOY / "champion.tflite")
shutil.copy("models/champion.json", DEPLOY / "champion.json")
print(f"Copied model + metadata to {DEPLOY.as_posix()}/")
