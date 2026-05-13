"""
IMU data normalizer — framework agnostic.

Use the same fitted scalers for both Keras and PyTorch pipelines.
Critical for ESP32 deployment: you MUST save these scalers and apply the
same transformation at inference time, then INVERT on the device output.

USAGE PATTERN:
    # ONE-TIME: fit on training data
    normalizer = IMUNormalizer()
    normalizer.fit_from_files(TRAIN_FILES)
    normalizer.save("./scalers")

    # In Keras dataset loader / PyTorch Dataset:
    normalizer = IMUNormalizer.load("./scalers")
    features_norm = normalizer.transform_features(features)
    labels_norm   = normalizer.transform_labels(labels)

    # After model prediction (to get back real-world angles):
    pred_real = normalizer.inverse_transform_labels(pred_norm)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


LABEL_COLS = ["Euler_x", "Euler_y", "Euler_z", "Quat_0", "Quat_1", "Quat_2", "Quat_3"]
EULER_COLS = ["Euler_x", "Euler_y", "Euler_z"]
QUAT_COLS = ["Quat_0", "Quat_1", "Quat_2", "Quat_3"]


class IMUNormalizer:
    """
    Handles normalization for IMU features + labels.

    Strategy:
      - Features (gyro, accel): StandardScaler (z-score)
      - Euler angles: StandardScaler in degrees (assumes no ±180° wrapping)
      - Quaternions: unit-norm enforcement only (no scaling — they're already
        in [-1, 1] and live on a hypersphere; scaling breaks the geometry)
    """

    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.euler_scaler = StandardScaler()
        self.feature_cols: list[str] | None = None
        self._fitted = False

    # ---------- fitting ----------

    def fit_from_files(self, file_paths: list[str]):
        """Fit scalers using ALL of the rows from the given CSV files."""
        dfs = [pd.read_csv(f) for f in file_paths]
        df = pd.concat(dfs, ignore_index=True)

        self.feature_cols = [c for c in df.columns if c not in LABEL_COLS]
        features = df[self.feature_cols].values.astype("float32")
        euler = df[EULER_COLS].values.astype("float32")
        quats = df[QUAT_COLS].values.astype("float32")

        return self.fit(features, euler, quats)

    def fit(self, features: np.ndarray, euler: np.ndarray, quats: np.ndarray):
        self.feature_scaler.fit(features)
        self.euler_scaler.fit(euler)

        # Sanity check on Euler wrapping
        if (euler.max(axis=0) > 175).any() and (euler.min(axis=0) < -175).any():
            print("  ⚠  Euler angles approach ±180° — wrapping may distort MSE.")
            print("     Consider sin/cos encoding instead.")

        # Sanity check on quaternion norms
        norms = np.linalg.norm(quats, axis=1)
        max_dev = np.abs(norms - 1.0).max()
        if max_dev > 0.01:
            print(f"  ⚠  Quaternion norms deviate from 1.0 (max deviation {max_dev:.4f}).")
            print("     Will normalize to unit norm during transform.")

        self._fitted = True
        return self

    # ---------- transform ----------

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() or load() first"
        return self.feature_scaler.transform(features).astype("float32")

    def transform_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Input  : (N, 7) raw labels in column order EULER_COLS + QUAT_COLS
        Output : (N, 7) normalized labels — Eulers z-scored, quats unit-norm
        """
        assert self._fitted
        euler = labels[:, :3]
        quats = labels[:, 3:]

        euler_norm = self.euler_scaler.transform(euler)
        # Enforce unit-norm on quaternions
        q_norms = np.linalg.norm(quats, axis=1, keepdims=True)
        q_norms = np.where(q_norms < 1e-8, 1.0, q_norms)  # avoid div-by-zero
        quats_norm = quats / q_norms

        return np.concatenate([euler_norm, quats_norm], axis=1).astype("float32")

    # ---------- inverse transform ----------

    def inverse_transform_labels(self, labels_norm: np.ndarray) -> np.ndarray:
        """
        Given model output (N, 7) of normalized predictions, convert back
        to raw units: Eulers in degrees, quaternions re-normalized to unit norm.
        """
        assert self._fitted
        euler_norm = labels_norm[:, :3]
        quats_norm = labels_norm[:, 3:]

        euler_deg = self.euler_scaler.inverse_transform(euler_norm)
        # Re-normalize quaternions (model output won't naturally be unit norm)
        q_norms = np.linalg.norm(quats_norm, axis=1, keepdims=True)
        q_norms = np.where(q_norms < 1e-8, 1.0, q_norms)
        quats_unit = quats_norm / q_norms

        return np.concatenate([euler_deg, quats_unit], axis=1)

    # ---------- persistence ----------

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.feature_scaler, os.path.join(directory, "feature_scaler.pkl"))
        joblib.dump(self.euler_scaler, os.path.join(directory, "euler_scaler.pkl"))
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump({
                "feature_cols": self.feature_cols,
                "feature_mean": self.feature_scaler.mean_.tolist(),
                "feature_scale": self.feature_scaler.scale_.tolist(),
                "euler_mean": self.euler_scaler.mean_.tolist(),
                "euler_scale": self.euler_scaler.scale_.tolist(),
            }, f, indent=2)
        print(f"  Scalers saved to {directory}/")
        print(f"  Feature mean: {self.feature_scaler.mean_}")
        print(f"  Feature std:  {self.feature_scaler.scale_}")

    @classmethod
    def load(cls, directory: str) -> "IMUNormalizer":
        normalizer = cls()
        normalizer.feature_scaler = joblib.load(
            os.path.join(directory, "feature_scaler.pkl")
        )
        normalizer.euler_scaler = joblib.load(
            os.path.join(directory, "euler_scaler.pkl")
        )
        with open(os.path.join(directory, "metadata.json")) as f:
            meta = json.load(f)
        normalizer.feature_cols = meta.get("feature_cols")
        normalizer._fitted = True
        return normalizer


# ============================================================
# One-shot script: run this ONCE before training to fit scalers
# ============================================================

if __name__ == "__main__":
    TRAIN_FILES = [
        "./dataset/IMU_Data_1.csv",
        "./dataset/IMU_Data_2.csv",
        "./dataset/IMU_Data_3.csv",
    ]
    OUTPUT_DIR = "./scalers"

    print("Fitting IMU normalizer on training data...")
    normalizer = IMUNormalizer()
    normalizer.fit_from_files(TRAIN_FILES)
    normalizer.save(OUTPUT_DIR)
    print("\n✓ Done. Use IMUNormalizer.load('./scalers') in your dataset loaders.")