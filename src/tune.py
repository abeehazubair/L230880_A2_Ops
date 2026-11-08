"""Task 2.7 apply - Keras Tuner random search on the VisionGuard MobileNetV2 model (use a GPU).
    python src/tune.py --trials 10
"""
import argparse
from pathlib import Path

import keras_tuner as kt
import mlflow
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv

from data import make_dataset
from models import build_transfer, unfreeze_last_block

p = argparse.ArgumentParser()
p.add_argument("--trials", type=int, default=10)
p.add_argument("--stage1-epochs", type=int, default=15)
p.add_argument("--stage2-epochs", type=int, default=15)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
load_dotenv()
tf.keras.utils.set_random_seed(args.seed)


class VisionGuardHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_transfer("mobilenetv2",
                              dropout=hp.Choice("dropout", [0.1, 0.2, 0.3, 0.4, 0.5]),
                              learning_rate=hp.Choice("learning_rate", [1e-3, 3e-4, 1e-4]))

    def fit(self, hp, model, *fit_args, callbacks=None, **kwargs):
        n_unfrozen = hp.Choice("unfrozen_layers", [0, 11, 30, 60])      # 11 = exactly the last block
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        cbs = (callbacks or []) + [stop]
        h1 = model.fit(*fit_args, epochs=args.stage1_epochs, callbacks=cbs, **kwargs)
        hist = {k: list(v) for k, v in h1.history.items()}
        if n_unfrozen:                                                   # stage 2: fine-tune
            unfreeze_last_block(model, hp.get("learning_rate") / 100, n_layers=n_unfrozen)
            start = len(hist["loss"])
            h2 = model.fit(*fit_args, initial_epoch=start, epochs=start + args.stage2_epochs,
                           callbacks=cbs, **kwargs)
            for k, v in h2.history.items():
                hist.setdefault(k, []).extend(v)
        return {"val_auc": max(hist["val_auc"]), "val_loss": min(hist["val_loss"])}


tuner = kt.RandomSearch(VisionGuardHyperModel(), objective=kt.Objective("val_auc", "max"),
                        max_trials=args.trials, seed=args.seed, directory="tuner_runs",
                        project_name="visionguard_mnv2", overwrite=True)
tuner.search_space_summary()
tuner.search(make_dataset("train"), validation_data=make_dataset("val"), verbose=0)

rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_auc": t.score}
        for t in tuner.oracle.get_best_trials(args.trials)]
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\nAll trials, best first:\n" + df.round(4).to_string(index=False))
out = Path("reports/tuner_visionguard_results.csv")
df.to_csv(out, index=False)

print("\nWhich setting mattered most? Mean val_auc per value:")
for name in ["learning_rate", "dropout", "unfrozen_layers"]:
    g = df.groupby(name).val_auc.agg(["mean", "count"])
    print(f"\n{name}  (spread between best and worst mean = {g['mean'].max() - g['mean'].min():.3f})")
    print(g.round(4).to_string())

best = tuner.get_best_hyperparameters(1)[0].values
print("\nBest settings:", best)

mlflow.set_experiment("VisionGuard")
with mlflow.start_run(run_name="keras_tuner_mobilenetv2"):
    mlflow.set_tags({"task": "2.7 keras tuner", "author": "L230880", "search": "RandomSearch"})
    mlflow.log_params({"trials": args.trials, "objective": "val_auc",
                       **{f"best_{k}": v for k, v in best.items()}})
    mlflow.log_metric("best_val_auc", float(df.val_auc.max()))
    mlflow.log_artifact(str(out))
    for r in df.itertuples():
        with mlflow.start_run(run_name=f"tuner_trial_{r.trial}", nested=True):
            mlflow.log_params({"learning_rate": r.learning_rate, "dropout": r.dropout,
                               "unfrozen_layers": r.unfrozen_layers})
            mlflow.log_metric("val_auc", float(r.val_auc))
print("Logged to MLflow run 'keras_tuner_mobilenetv2'")
