"""Task 3.1 / 3.2 - MLflow tool lab on Fashion-MNIST (a built-in Keras dataset).
    python labs/mlflow_toy.py --lr 0.001
    python labs/mlflow_toy.py --lr 0.01
Without a .env file runs are stored locally in ./mlruns (view with: mlflow ui).
With a .env file holding MLFLOW_TRACKING_URI/USERNAME/PASSWORD they go to DagsHub (Task 3.2).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()                                   # reads .env if it exists (used from Task 3.2 on)

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--units", type=int, default=64)
args = p.parse_args()

tf.keras.utils.set_random_seed(42)
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
x_tr, y_tr = x_tr[:20000] / 255.0, y_tr[:20000]          # subset keeps the toy fast on CPU
x_te = x_te / 255.0

model = tf.keras.Sequential([
    tf.keras.Input((28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(args.units, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(tf.keras.optimizers.Adam(args.lr), "sparse_categorical_crossentropy", ["accuracy"])


class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Sends loss/accuracy (train and val) to MLflow after every epoch."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


mlflow.set_experiment("fashion_mnist_toy")
with mlflow.start_run(run_name=f"lr={args.lr}"):
    mlflow.set_tags({"task": "3.1 tool lab", "dataset": "fashion_mnist", "author": "L230880"})
    mlflow.log_params({"learning_rate": args.lr, "epochs": args.epochs, "units": args.units,
                       "batch_size": 128, "train_samples": len(x_tr)})

    hist = model.fit(x_tr, y_tr, validation_split=0.2, epochs=args.epochs, batch_size=128,
                     callbacks=[MLflowEpochLogger()], verbose=2)

    test_loss, test_acc = model.evaluate(x_te, y_te, verbose=0)
    mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m in zip(axes, ["loss", "accuracy"]):
        ax.plot(hist.history[m], label="train")
        ax.plot(hist.history["val_" + m], label="validation")
        ax.set_title(f"{m} (lr={args.lr})")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    mlflow.log_figure(fig, "training_curves.png")        # the plot artifact

    mlflow.tensorflow.log_model(model, artifact_path="model")   # the model artifact
    print(f"\ntest_accuracy={test_acc:.4f}   run_id={mlflow.active_run().info.run_id}")
    print("tracking URI:", mlflow.get_tracking_uri())
