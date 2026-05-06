# Requires activating the torch_env conda environmentrun
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

class IMUDataset(Dataset):
    def __init__(self, imu_file: list[str], sequence_length: int = 10):
        self.sequence_length = sequence_length
        
        df_temp = [pd.read_csv(file) for file in imu_file]
        df = pd.concat(df_temp, ignore_index=True)

        label_cols = ["Euler_x", "Euler_y", "Euler_z", "Quat_0", "Quat_1", "Quat_2", "Quat_3"]
        feature_cols = [col for col in df.columns if col not in label_cols]

        features_np = df[feature_cols].values.astype('float32')
        labels_np = df[label_cols].values.astype('float32')

        features = torch.from_numpy(features_np) # (Total_Rows, 9)
        
        # --- THE MAGIC TRICK ---
        # 1. Transpose to (9, Total_Rows)
        features = features.transpose(0, 1)
        
        # 2. Unfold creates sliding windows over the Total_Rows dimension
        # Resulting shape: (9, Num_Samples, sequence_length)
        features = features.unfold(dimension=1, size=sequence_length, step=1)
        
        # 3. Permute to get it into (Num_Samples, 9, sequence_length)
        features = features.permute(1, 0, 2)
        
        # 4. Unsqueeze to get final shape: (Num_Samples, 9, sequence_length, 1)
        self.x_data = features.unsqueeze(-1).contiguous()
        
        # Keep labels the same
        self.y_data = torch.from_numpy(labels_np[sequence_length - 1:])
        
    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        # Zero math, zero transposing. Just an O(1) memory lookup.
        return self.x_data[idx], self.y_data[idx]
# class IMUDataset(Dataset):
#     def __init__(self, imu_file: list[str], sequence_length: int = 10):
#         self.sequence_length = sequence_length
#         # Load Dataset into numpy arrays
#         df_temp = []
#         for file in imu_file:
#             df_temp.append(pd.read_csv(file))
            
#         df = pd.concat(df_temp, ignore_index=True)

#         label_cols = ["Euler_x", "Euler_y", "Euler_z", "Quat_0", "Quat_1", "Quat_2", "Quat_3"]
#         feature_cols = [col for col in df.columns if col not in label_cols]

        
#         self.features = torch.from_numpy(df[feature_cols].values.astype('float32'))
#         labels = df[label_cols].values.astype('float32')

#         self.labels = torch.from_numpy(labels[sequence_length - 1:])
        
#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # Format as (N, Cin, H, W) from the raw data in features formated as (dataset_size, 10, 9)
#         # Slice from idx to idx+ sequence_length to get x value at this point
#         x = torch.unsqueeze(torch.transpose(self.features[idx: idx + self.sequence_length, :], 0, 1), -1)
#         y = self.labels[idx, :]
#         return x, y
    
if __name__ == "__main__":
    data = IMUDataset(["./dataset/IMU_Data_1.csv", "./dataset/IMU_Data_2.csv"], 10)
    batched_data = DataLoader(dataset=data, batch_size=32)

    for x, y in batched_data:
        print(f"X: {x.shape}, Y: {y.shape}")
        break