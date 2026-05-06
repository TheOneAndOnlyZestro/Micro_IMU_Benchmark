import pandas as pd
import tensorflow as tf

import keras
def load_dataset(imu_file: list[str], batch_size: int = 32) -> tf.data.Dataset:
    df_temp = []
    for file in imu_file:
        df_temp.append(pd.read_csv(file))
        
    df = pd.concat(df_temp, ignore_index=True)
    print(df.head())
    columns = ["Euler_x","Euler_y","Euler_z","Quat_0","Quat_1","Quat_2","Quat_3"]
    df_labels = df[columns]
    df_features = df[list(set(df.columns.tolist()) - set(columns))]

    
    features = tf.convert_to_tensor(df_features.values, dtype=tf.float32)
    labels = tf.convert_to_tensor(df_labels.values, dtype=tf.float32)

    print(f"Loaded {features.shape[0]} rows with {features.shape[1]} features")
    imu_slices = tf.data.Dataset.from_tensor_slices((features, labels))

    imu_slices = imu_slices.batch(batch_size=batch_size)

    for f, l in imu_slices:
        print(f)
        print(f"Features: {f.shape}")
        break

    imu_slices = imu_slices.cache().prefetch(tf.data.AUTOTUNE)
    return imu_slices

def load_dataset_conv(imu_file: list[str], sequence_length: int = 10, batch_size: int = 32) -> tf.data.Dataset:
    df_temp = []
    for file in imu_file:
        df_temp.append(pd.read_csv(file))
        
    df = pd.concat(df_temp, ignore_index=True)

    label_cols = ["Euler_x", "Euler_y", "Euler_z", "Quat_0", "Quat_1", "Quat_2", "Quat_3"]
    feature_cols = [col for col in df.columns if col not in label_cols]

    features = df[feature_cols].values.astype('float32')
    labels = df[label_cols].values.astype('float32')

    targets_aligned = labels[sequence_length - 1:]

    # Create the rolling window dataset
    dataset = keras.utils.timeseries_dataset_from_array(
        data=features,
        targets=targets_aligned,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False 
    )

    # x shape goes from (Batch, Seq, 9) -> (Batch, Seq, 1, 9)
    dataset = dataset.map(lambda x, y: (tf.expand_dims(x, axis=2), y))

    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
    return dataset