"""Task 2.2 / 2.3 - model definitions for VisionGuard.
    python src/models.py            -> baseline CNN summary + per-layer table
    python src/models.py transfer   -> MobileNetV2 stages: trainable layers in each stage
"""
import sys
from pathlib import Path

import numpy as np
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


# ---------------------------------------------------------------- 2.2 baseline CNN
def conv_block(x, filters, name):
    """Conv -> BatchNorm -> ReLU -> MaxPool: find patterns, stabilise, keep strong ones, shrink."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn")(x)

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


# ---------------------------------------------------------------- 2.3 transfer learning
# backbone -> (keras.applications class, its own preprocessing as a saveable layer, first layer of last block)
BACKBONES = {
    # MobileNetV2 preprocess_input = x/127.5 - 1  (pixels 0..255 -> -1..1), same as this Rescaling
    "mobilenetv2": (tf.keras.applications.MobileNetV2,
                    lambda: layers.Rescaling(1 / 127.5, offset=-1.0, name="preprocess"),
                    "block_16_expand"),
    # EfficientNetB0 rescales internally: its preprocess_input does nothing
    "efficientnetb0": (tf.keras.applications.EfficientNetB0,
                       lambda: layers.Identity(name="preprocess"),
                       "block7a_expand_conv"),
}


def build_transfer(backbone="mobilenetv2", img_size=IMG_SIZE, dropout=0.3, learning_rate=1e-3):
    """Stage 1: pretrained backbone without its classifier, frozen, plus our own head."""
    app, preprocess_layer, _ = BACKBONES[backbone]
    base = app(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
    base.trainable = False                                   # freeze all pretrained weights

    inputs = tf.keras.Input((img_size, img_size, 3), name="image")       # raw pixels 0-255
    x = preprocess_layer()(inputs)                           # exactly what the backbone expects
    x = base(x, training=False)                              # BatchNorm stays in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return compile_binary(tf.keras.Model(inputs, outputs, name=f"{backbone}_transfer"), learning_rate)


def get_backbone(model):
    return next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))


def unfreeze_last_block(model, learning_rate=1e-5, n_layers=None):
    """Stage 2: unfreeze only the last block (or the last n_layers) and recompile with a small lr."""
    backbone = model.name.replace("_transfer", "")
    base = get_backbone(model)
    base.trainable = True
    names = [layer.name for layer in base.layers]
    start = len(names) - n_layers if n_layers else names.index(BACKBONES[backbone][2])
    for i, layer in enumerate(base.layers):
        # BatchNorm layers stay frozen: their statistics come from 1.4M ImageNet images
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
    return compile_binary(model, learning_rate)              # recompile so the change takes effect


def trainable_summary(model):
    base = get_backbone(model)
    inside = [layer.name for layer in base.layers if layer.trainable and layer.trainable_weights]
    head = [layer.name for layer in model.layers if layer is not base and layer.trainable_weights]
    n_train = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    return inside, head, n_train, model.count_params()


def layer_table(model):
    rows = []
    for layer in model.layers:
        shape = layer.output_shape
        shape = shape[0] if isinstance(shape, list) else shape
        rows.append({"layer": layer.name, "type": type(layer).__name__,
                     "output_shape": str(shape), "params": layer.count_params()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "transfer":
        model = build_transfer("mobilenetv2")
        x = tf.random.uniform((2, IMG_SIZE, IMG_SIZE, 3), 0, 255)
        same = np.allclose(model.get_layer("preprocess")(x),
                           tf.keras.applications.mobilenet_v2.preprocess_input(tf.identity(x)), atol=1e-5)
        print(f"Our preprocess layer == mobilenet_v2.preprocess_input: {same}")
        print(f"Backbone has {len(get_backbone(model).layers)} layers\n")

        lines = []
        for stage, title in [(1, "STAGE 1 - backbone frozen, lr=1e-3"),
                             (2, "STAGE 2 - last block unfrozen, lr=1e-5")]:
            if stage == 2:
                unfreeze_last_block(model)
            inside, head, n_train, n_total = trainable_summary(model)
            lines += [f"== {title} ==",
                      f"Trainable params: {n_train:,} of {n_total:,} ({n_train / n_total:.1%})",
                      f"Head layers (trainable): {head}",
                      f"Backbone layers trainable ({len(inside)}): {inside}", ""]
        print("\n".join(lines))
        out = Path("reports/transfer_trainable_layers.txt")
        out.write_text("\n".join(lines))
        print("saved", out.as_posix())
    else:
        model = build_baseline_cnn()
        model.summary()
        table = layer_table(model)
        out = Path("reports/cnn_layers.csv")
        table.to_csv(out, index=False)
        print("\n" + table.to_string(index=False))
        print(f"\nTotal params: {table.params.sum():,}   saved {out.as_posix()}")
