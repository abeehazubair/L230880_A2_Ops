"""VisionGuard training with MLflow tracking (Tasks 2.3, 2.4, 2.5, 3.3, 3.4).

    python src/train.py --model cnn                     # manual logging (Keras callback)
    python src/train.py --model mobilenetv2             # frozen stage + fine-tune last block
    python src/train.py --model cnn --autolog           # MLflow TensorFlow autologging
Runs go to DagsHub when .env holds the MLFLOW_TRACKING_* variables.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from dotenv import load_dotenv

from data import make_dataset
from evaluate import (BLUE, GREY, ORANGE, choose_threshold, compute_metrics, plot_confusion,
                      plot_roc, predict)
from models import build_baseline_cnn, build_transfer, trainable_summary, unfreeze_last_block


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cnn", choices=["cnn", "mobilenetv2", "efficientnetb0"])
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate (stage 1 for transfer)")
    p.add_argument("--epochs", type=int, default=30, help="max epochs (stage 1 for transfer)")
    p.add_argument("--ft-lr", type=float, default=1e-5, help="fine-tuning learning rate (stage 2)")
    p.add_argument("--ft-epochs", type=int, default=15, help="max fine-tuning epochs (stage 2)")
    p.add_argument("--unfreeze-layers", type=int, default=None, help="default: the last block")
    p.add_argument("--no-finetune", action="store_true", help="transfer: skip stage 2")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--autolog", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--experiment", default="VisionGuard")
    return p.parse_args()


def git_info():
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))


def dvc_data_md5():
    return yaml.safe_load(Path("data/raw.dvc").read_text())["outs"][0]["md5"]


class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Manual logging: train/val metrics and the current learning rate after every epoch."""
    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in (logs or {}).items()}
        logs["lr"] = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        mlflow.log_metrics(logs, step=epoch)


def fit_stage(model, train_ds, val_ds, start, epochs, ckpt, patience, manual, history, threshold=None):
    """One training stage with the 2.4 callbacks. Ends with the best (lowest val_loss) weights loaded."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True,
                                           save_weights_only=True, initial_value_threshold=threshold,
                                           verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=max(2, patience // 3), min_lr=1e-7, verbose=1),
    ]
    if manual:
        callbacks.append(MLflowEpochLogger())
    h = model.fit(train_ds, validation_data=val_ds, initial_epoch=start, epochs=start + epochs,
                  callbacks=callbacks, verbose=2)
    for k, v in h.history.items():
        history.setdefault(k, []).extend(float(x) for x in v)
    if ckpt.exists():
        model.load_weights(str(ckpt))          # roll back to this stage's best epoch
    return ckpt.exists()


def plot_curves(history, best_epoch, boundary, title, path):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[m], color=BLUE, lw=2, label="train")
        ax.plot(epochs, history["val_" + m], color=ORANGE, lw=2, label="validation")
        ax.axvline(best_epoch, color=GREY, ls="--", lw=1.2, label=f"best epoch = {best_epoch}")
        if boundary:
            ax.axvline(boundary + 0.5, color=GREY, ls=":", lw=1.2, label="start fine-tuning")
        ax.set_xlabel("epoch")
        ax.set_title(f"{m} - {title}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    load_dotenv()
    tf.keras.utils.set_random_seed(args.seed)          # Python, NumPy and TensorFlow seeds
    if args.autolog and args.model != "cnn":
        sys.exit("--autolog is meant for the single-stage CNN run (one fit call)")
    transfer = args.model != "cnn"
    dropout = args.dropout if args.dropout is not None else (0.3 if transfer else 0.4)
    augment = not args.no_augment
    name = args.run_name or f"{args.model}_lr{args.lr:g}" + ("_autolog" if args.autolog else "")
    out = Path("outputs") / name
    out.mkdir(parents=True, exist_ok=True)

    train_ds = make_dataset("train", args.img_size, args.batch_size, augment=augment, seed=args.seed)
    val_ds = make_dataset("val", args.img_size, args.batch_size)
    test_ds = make_dataset("test", args.img_size, args.batch_size)

    commit, dirty = git_info()
    mlflow.set_experiment(args.experiment)
    if args.autolog:
        mlflow.tensorflow.autolog(log_models=True, log_datasets=False)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"git_commit": commit, "git_dirty": str(dirty), "dvc_data_md5": dvc_data_md5(),
                         "seed": str(args.seed), "model_type": args.model, "author": "L230880",
                         "logging": "autolog" if args.autolog else "manual",
                         "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU"})
        params = {"model_type": args.model, "img_size": args.img_size, "augment": augment,
                  "dropout": dropout, "seed": args.seed, "patience": args.patience,
                  "train_images": len(pd.read_csv("splits/train.csv"))}
        if not args.autolog:     # autolog records its own optimizer/epoch params
            params.update(learning_rate=args.lr, max_epochs=args.epochs, batch_size=args.batch_size)
        if transfer:
            params.update(finetune=not args.no_finetune, ft_learning_rate=args.ft_lr,
                          ft_max_epochs=args.ft_epochs,
                          unfreeze=args.unfreeze_layers or "last_block")
        mlflow.log_params(params)
        if not transfer:
            mlflow.log_param("bn_momentum", 0.9)

        history, boundary, manual = {}, None, not args.autolog
        if not transfer:
            model = build_baseline_cnn(args.img_size, dropout, args.lr)
            fit_stage(model, train_ds, val_ds, 0, args.epochs, out / "best.weights.h5",
                      args.patience, manual, history)
        else:
            model = build_transfer(args.model, args.img_size, dropout, args.lr)
            mlflow.log_param("stage1_trainable_params", trainable_summary(model)[2])
            fit_stage(model, train_ds, val_ds, 0, args.epochs, out / "stage1.weights.h5",
                      args.patience, manual, history)
            boundary = len(history["loss"])
            if not args.no_finetune:
                unfreeze_last_block(model, args.ft_lr, args.unfreeze_layers)
                inside, _, n_train, _ = trainable_summary(model)
                mlflow.log_params({"stage2_trainable_params": n_train,
                                   "stage2_unfrozen_layers": ",".join(inside)[:500]})
                improved = fit_stage(model, train_ds, val_ds, boundary, args.ft_epochs,
                                     out / "stage2.weights.h5", args.patience, manual, history,
                                     threshold=min(history["val_loss"]))
                if not improved:
                    print("Fine-tuning never beat stage 1 -> keeping the stage-1 weights")
                    model.load_weights(str(out / "stage1.weights.h5"))

        best_idx = int(np.argmin(history["val_loss"]))
        best_epoch = best_idx + 1
        mlflow.log_metrics({"best_epoch": best_epoch, "best_val_loss": history["val_loss"][best_idx],
                            "epochs_trained": len(history["loss"])})

        # ---- 2.5: threshold from validation, test scored once with it
        val_true, val_prob = predict(model, val_ds)
        thr = choose_threshold(val_true, val_prob)
        test_true, test_prob = predict(model, test_ds)
        vm, tm = compute_metrics(val_true, val_prob, thr), compute_metrics(test_true, test_prob, thr)
        mlflow.log_metric("threshold", thr)
        mlflow.log_metrics({f"final_val_{k}": v for k, v in vm.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in tm.items()})

        # ---- artifacts (3.4)
        plot_curves(history, best_epoch, boundary, name, out / "training_curves.png")
        plot_confusion(tm, f"Test confusion matrix - {name} (thr={thr:.2f})", out / "confusion_matrix_test.png")
        plot_roc(test_true, test_prob, thr, f"Test ROC - {name}", out / "roc_test.png")
        pd.read_csv("splits/test.csv").assign(p_ai=test_prob, pred=(test_prob >= thr).astype(int)) \
          .to_csv(out / "test_predictions.csv", index=False)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        (out / "requirements-freeze.txt").write_text(freeze.stdout)
        model.save(str(out / "model.keras"))
        summary = {"run_id": run.info.run_id, "best_epoch": best_epoch, "threshold": thr,
                   "val": vm, "test": tm}
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

        for f in ["training_curves.png", "confusion_matrix_test.png", "roc_test.png",
                  "test_predictions.csv", "summary.json"]:
            mlflow.log_artifact(str(out / f), artifact_path="evaluation")
        mlflow.log_artifact(str(out / "requirements-freeze.txt"), artifact_path="environment")
        mlflow.log_artifacts("splits", artifact_path="splits")
        mlflow.log_artifact(str(out / "model.keras"), artifact_path="keras_model")
        if not args.autolog:                 # autolog already logs the model itself
            mlflow.tensorflow.log_model(model, artifact_path="model")

        print(f"\n=== {name} ===  run_id={run.info.run_id}")
        print(f"best epoch {best_epoch} of {len(history['loss'])}, threshold {thr:.2f} (chosen on val)")
        print("test:", {k: round(v, 4) if isinstance(v, float) else v for k, v in tm.items()})


if __name__ == "__main__":
    main()
