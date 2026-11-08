"""Task 3.6 - load the VisionGuard model BY ALIAS from the registry and classify one image.
    python src/predict_champion.py path/to/image.jpg
"""
import sys

import mlflow
import tensorflow as tf
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from data import decode_and_resize            # the exact preprocessing used in training

load_dotenv(".env")
NAME, ALIAS = "VisionGuard", "champion"
image_path = sys.argv[1]

client = MlflowClient()
mv = client.get_model_version_by_alias(NAME, ALIAS)
threshold = float(mv.tags["threshold"])
img_size = int(mv.tags["img_size"])
print(f"models:/{NAME}@{ALIAS} -> version {mv.version} (run {mv.tags['run_name']}), "
      f"threshold {threshold}, image size {img_size}")

model = mlflow.tensorflow.load_model(f"models:/{NAME}@{ALIAS}")
x = tf.cast(decode_and_resize(tf.io.read_file(image_path), img_size), tf.float32)[tf.newaxis]
p_ai = float(model.predict(x, verbose=0).ravel()[0])

label = "AI-generated" if p_ai >= threshold else "Real"
confidence = p_ai if label == "AI-generated" else 1 - p_ai
print(f"\nImage: {image_path}\nP(AI) = {p_ai:.4f}  ->  {label}  (confidence {confidence:.1%})")
