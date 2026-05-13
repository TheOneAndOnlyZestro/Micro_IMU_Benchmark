"""
Keras IMU dataset loader with applied normalization.

The normalization is done in NumPy (eager) before the tf.data pipeline,
so the dataset just streams pre-normalized values. This is the cleanest
approach — keeping scaler logic out of the TF graph means we never have
to worry about TF graph-mode quirks with sklearn objects.
"""

import keras
import pandas as pd
import tensorflow as tf

from Imu_normalizer import IMUNormalizer, LABEL_COLS


def load_dataset_conv(
    imu_file: list[str],
    sequence_length: int = 10,
    batch_size: int = 32,
    normalizer: IMUNormalizer | None = None,
    scaler_dir: str = "./scalers",
) -> tf.data.Dataset:
    """
    Returns a tf.data.Dataset yielding (x, y) where:
        x : (Batch, Seq, 1, 9) float32, normalized features
        y : (Batch, 7) float32, normalized labels
    """
    if normalizer is None:
        normalizer = IMUNormalizer.load(scaler_dir)

    df_temp = [pd.read_csv(f) for f in imu_file]
    df = pd.concat(df_temp, ignore_index=True)

    feature_cols = [col for col in df.columns if col not in LABEL_COLS]
    features = df[feature_cols].values.astype("float32")
    labels = df[LABEL_COLS].values.astype("float32")

    # APPLY NORMALIZATION EAGERLY — before tf.data
    features = normalizer.transform_features(features)
    labels = normalizer.transform_labels(labels)

    targets_aligned = labels[sequence_length - 1:]

    dataset = keras.utils.timeseries_dataset_from_array(
        data=features,
        targets=targets_aligned,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
    )

    # (Batch, Seq, 9) -> (Batch, Seq, 1, 9)
    dataset = dataset.map(lambda x, y: (tf.expand_dims(x, axis=2), y))

    return dataset.cache().prefetch(tf.data.AUTOTUNE)