"""Task 2.2 / 2.3 - model definitions for VisionGuard.
Run directly to print the baseline CNN summary and its per-layer table:
    python src/models.py
"""
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = 224


def compile_binary(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


def conv_block(x, filters, name):
    """Conv -> BatchNorm -> ReLU -> MaxPool: find patterns, stabilise, keep strong ones, shrink."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return layers.MaxPooling2D(2, name=f"{name}_pool")(x)


def build_baseline_cnn(img_size=IMG_SIZE, dropout=0.4, learning_rate=1e-3):
    inputs = tf.keras.Input((img_size, img_size, 3), name="image")       # raw pixels 0-255
    x = layers.Rescaling(1.0 / 255, name="rescale_0_1")(inputs)          # scaling lives in the model
    for i, filters in enumerate((32, 64, 128, 256), start=1):
        x = conv_block(x, filters, f"block{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)      # P(image is AI-generated)
    return compile_binary(tf.keras.Model(inputs, outputs, name="baseline_cnn"), learning_rate)


def layer_table(model):
    rows = []
    for layer in model.layers:
        shape = layer.output_shape
        shape = shape[0] if isinstance(shape, list) else shape
        rows.append({"layer": layer.name, "type": type(layer).__name__,
                     "output_shape": str(shape), "params": layer.count_params()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    model = build_baseline_cnn()
    model.summary()
    table = layer_table(model)
    out = Path("reports/cnn_layers.csv")
    table.to_csv(out, index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nTotal params: {table.params.sum():,}   saved {out.as_posix()}")
