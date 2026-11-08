"""Task 2.1 - tf.data input pipeline for VisionGuard.

Imported by every training script; decode_and_resize() is also reused by the
Flask app so that training and serving preprocessing are identical.
Run directly to check the pipeline and save a figure of one augmented batch:
    python src/data.py
"""
from pathlib import Path

import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
SPLITS = Path("splits")
CLASS_NAMES = ["RealArt", "AiArtData"]          # index = label, 1 = AI-generated


def decode_and_resize(image_bytes, img_size=IMG_SIZE):
    """Raw file bytes -> uint8 tensor of shape (img_size, img_size, 3)."""
    # decode_image reads the real format from the file content (JPEG/PNG/GIF/BMP),
    # so wrong extensions don't matter; channels=3 turns L, P and RGBA into RGB.
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    # Scale the longer side to img_size and pad the rest with black:
    # keeps the aspect ratio instead of stretching the picture.
    img = tf.image.resize_with_pad(img, img_size, img_size, antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def _load(path, label, img_size):
    return decode_and_resize(tf.io.read_file(path), img_size), label


def build_augmenter(seed=SEED):
    """Small random, label-preserving changes - used on TRAINING batches only."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.05, fill_mode="constant", seed=seed),
        tf.keras.layers.RandomZoom(0.10, fill_mode="constant", seed=seed),
    ], name="augmentation")


def make_dataset(split, img_size=IMG_SIZE, batch_size=BATCH_SIZE, augment=None, seed=SEED):
    """split = 'train' | 'val' | 'test'. Returns batches of (float32 images 0-255, labels)."""
    import pandas as pd          # training-only dependency; the serving image does not need pandas
    df = pd.read_csv(SPLITS / f"{split}.csv")
    training = split == "train"
    augment = training if augment is None else (augment and training)   # never augment val/test

    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label.values.astype("float32")))
    ds = ds.map(lambda p, y: _load(p, y, img_size), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()                                  # decode each image once, reuse every epoch
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if augment:
        aug = build_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)                     # prepare next batch while the model trains


if __name__ == "__main__":
    import time

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf.keras.utils.set_random_seed(SEED)
    datasets = {}
    for split in ("train", "val", "test"):
        datasets[split] = make_dataset(split)
        t = time.time()
        n = sum(int(y.shape[0]) for _, y in datasets[split])
        print(f"{split:5s}: {n} images, first pass (decode + resize) {time.time() - t:.1f}s")

    x, y = next(iter(datasets["train"]))
    print("batch images:", x.shape, x.dtype,
          f"min={float(tf.reduce_min(x)):.0f} max={float(tf.reduce_max(x)):.0f}")
    print("batch labels:", y.shape, y.dtype, y[:10].numpy())

    fig, axes = plt.subplots(4, 4, figsize=(12, 12.5))
    for ax, img, lab in zip(axes.flat, x[:16], y[:16]):
        ax.imshow(img.numpy().astype("uint8"))
        ax.set_title(CLASS_NAMES[int(lab)], fontsize=11)
        ax.axis("off")
    fig.suptitle(f"One augmented training batch ({IMG_SIZE}x{IMG_SIZE}, padded to keep "
                 "aspect ratio; flip + rotate ±18° + zoom ±10%)", fontsize=12)
    plt.tight_layout()
    out = Path("reports/figures/fig_augmented_batch.png")
    fig.savefig(out, dpi=120)
    print("Saved", out.as_posix())
