"""Task 7 - does the cloud model give the same answers as the original?

Changing the runtime (Keras -> TFLite) and the preprocessing (TensorFlow -> Pillow/NumPy) is a change
to the serving path, i.e. a possible source of training/serving skew (4.1). On every TEST image this
compares three ways of computing P(AI):
  A  Keras SavedModel + TF preprocessing        (what we trained and evaluated)
  B  TFLite            + TF preprocessing        (isolates the conversion)
  C  TFLite            + Pillow/NumPy preproc.   (exactly what the Vercel API computes)
    python src/check_tflite_parity.py      -> reports/tflite_parity.csv
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import decode_and_resize  # noqa: E402

os.environ["API_KEY"] = os.environ.get("API_KEY") or "local-parity-check"   # the API module refuses to load without one
sys.path.insert(0, str(Path("deploy/vercel/api").resolve()))
from index import decode_resize_np  # noqa: E402

meta = json.loads(Path("models/champion.json").read_text())
thr, size = meta["threshold"], meta["img_size"]
keras_model = tf.keras.models.load_model("models/champion", compile=False)
lite = tf.lite.Interpreter(model_path="models/champion.tflite")
lite.allocate_tensors()
li, lo = lite.get_input_details()[0]["index"], lite.get_output_details()[0]["index"]


def run_lite(x):
    lite.set_tensor(li, x)
    lite.invoke()
    return float(lite.get_tensor(lo).ravel()[0])


rows = []
test = pd.read_csv("splits/test.csv")
for r in test.itertuples():
    data = Path(r.path).read_bytes()
    x_tf = tf.cast(decode_and_resize(tf.constant(data), size), tf.float32)[tf.newaxis].numpy()
    x_np = decode_resize_np(data, size)
    rows.append({"path": r.path, "label": r.label,
                 "p_keras": float(keras_model(x_tf, training=False).numpy().ravel()[0]),
                 "p_tflite_tfprep": run_lite(x_tf), "p_tflite_np": run_lite(x_np),
                 "max_pixel_diff": float(np.abs(x_tf - x_np).max())})
df = pd.DataFrame(rows)
Path("reports").mkdir(exist_ok=True)
df.to_csv("reports/tflite_parity.csv", index=False)

print(f"{len(df)} test images, threshold {thr}\n")
print(f"{'Comparison':58s} {'max |dp|':>10s} {'mean |dp|':>10s} {'same decision':>14s}")
for name, a, b in [("A vs B: Keras vs TFLite (same TF preprocessing)", "p_keras", "p_tflite_tfprep"),
                   ("A vs C: Keras+TF vs TFLite+Pillow/NumPy (Vercel)", "p_keras", "p_tflite_np")]:
    d = (df[a] - df[b]).abs()
    same = ((df[a] >= thr) == (df[b] >= thr)).sum()
    print(f"{name:58s} {d.max():10.2e} {d.mean():10.2e} {same:>7d} / {len(df)}")
for col, label in [("p_keras", "Keras (original)"), ("p_tflite_np", "TFLite on Vercel")]:
    print(f"Test accuracy {label:18s}: {((df[col] >= thr).astype(int) == df.label).mean():.4f}")

# Where does the preprocessing difference come from? Compare JPEG decoders on the test JPEGs.
diffs_fast, diffs_acc = [], []
for p in df.path:
    data = Path(p).read_bytes()
    if not data.startswith(b"\xff\xd8"):          # JPEG files only
        continue
    pil = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
    fast = tf.io.decode_jpeg(data, channels=3).numpy().astype(np.int16)                      # TF default
    acc = tf.io.decode_jpeg(data, channels=3, dct_method="INTEGER_ACCURATE").numpy().astype(np.int16)
    if fast.shape == pil.shape:
        diffs_fast.append(np.abs(fast - pil).max())
        diffs_acc.append(np.abs(acc - pil).max())
print(f"\nJPEG decoder check on {len(diffs_fast)} test JPEGs (max pixel difference vs Pillow):")
print(f"  TF default (fast integer DCT)     : up to {max(diffs_fast)} / 255")
print(f"  TF dct_method='INTEGER_ACCURATE'  : up to {max(diffs_acc)} / 255")
print("Saved reports/tflite_parity.csv")
