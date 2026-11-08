"""Task 2.7 toy - Keras Tuner random search on Fashion-MNIST.
    python labs/tuner_toy.py
"""
from pathlib import Path

import keras_tuner as kt
import pandas as pd
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
(x, y), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x, y, x_test = x[:20000] / 255.0, y[:20000], x_test / 255.0       # subset keeps it fast


def build(hp):
    units = hp.Int("units", min_value=32, max_value=256, step=32)
    lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    model = tf.keras.Sequential([
        tf.keras.Input((28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", ["accuracy"])
    return model


tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=10, seed=42,
                        directory="tuner_runs", project_name="fashion_toy", overwrite=True)
tuner.search_space_summary()
tuner.search(x, y, validation_split=0.2, epochs=5, batch_size=128, verbose=0)

rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": round(t.score, 4)}
        for t in tuner.oracle.get_best_trials(num_trials=10)]
results = pd.DataFrame(rows)
print("\nAll trials, best first:\n" + results.to_string(index=False))
Path("reports").mkdir(exist_ok=True)
results.to_csv("reports/tuner_toy_results.csv", index=False)

best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.get_best_models(1)[0]
print("\nBest settings:", best_hp.values)
print("Best model test accuracy:", round(best_model.evaluate(x_test, y_test, verbose=0)[1], 4))
