"""
PyTorch IMU dataset with applied normalization.

Loads pre-fitted scalers from disk and applies them eagerly during
__init__ so __getitem__ stays an O(1) lookup.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from Imu_normalizer import IMUNormalizer, LABEL_COLS


class IMUDataset(Dataset):
    def __init__(
        self,
        imu_file: list[str],
        sequence_length: int = 10,
        normalizer: IMUNormalizer | None = None,
        scaler_dir: str = "./scalers",
    ):
        """
        Args:
            imu_file: list of CSV paths to load
            sequence_length: rolling window size
            normalizer: optional pre-loaded IMUNormalizer; if None, loaded from scaler_dir
            scaler_dir: directory containing saved scalers (only used if normalizer is None)
        """
        self.sequence_length = sequence_length
        self.normalizer = normalizer or IMUNormalizer.load(scaler_dir)

        df_temp = [pd.read_csv(file) for file in imu_file]
        df = pd.concat(df_temp, ignore_index=True)

        feature_cols = [col for col in df.columns if col not in LABEL_COLS]
        features_np = df[feature_cols].values.astype("float32")
        labels_np = df[LABEL_COLS].values.astype("float32")

        # APPLY NORMALIZATION EAGERLY — before any windowing
        features_np = self.normalizer.transform_features(features_np)
        labels_np = self.normalizer.transform_labels(labels_np)

        features = torch.from_numpy(features_np)  # (Total_Rows, 9)

        # 1. Transpose to (9, Total_Rows)
        features = features.transpose(0, 1)

        # 2. Unfold over rows -> (9, Num_Samples, sequence_length)
        features = features.unfold(dimension=1, size=sequence_length, step=1)

        # 3. Permute -> (Num_Samples, 9, sequence_length)
        features = features.permute(1, 0, 2)

        # 4. Unsqueeze -> (Num_Samples, 9, sequence_length, 1)
        self.x_data = features.unsqueeze(-1).contiguous()

        # Labels are aligned with the END of each window
        self.y_data = torch.from_numpy(labels_np[sequence_length - 1:])

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


if __name__ == "__main__":
    # First make sure you've run `python imu_normalizer.py` to fit the scalers
    data = IMUDataset(
        ["./dataset/IMU_Data_1.csv", "./dataset/IMU_Data_2.csv"],
        sequence_length=10,
    )
    batched_data = DataLoader(dataset=data, batch_size=32)

    for x, y in batched_data:
        print(f"X: shape={tuple(x.shape)} mean={x.mean():.3f} std={x.std():.3f}")
        print(f"Y: shape={tuple(y.shape)} mean={y.mean():.3f} std={y.std():.3f}")
        break