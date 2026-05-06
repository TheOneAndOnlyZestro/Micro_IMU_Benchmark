import pandas as pd
import numpy as np

def generate_c_header(csv_file: str, output_h_file: str = "imu_dataset.h", sequence_length: int = 10, num_samples: int = 5):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 1. Separate features and labels
    label_cols =["Euler_x", "Euler_y", "Euler_z", "Quat_0", "Quat_1", "Quat_2", "Quat_3"]
    feature_cols =[col for col in df.columns if col not in label_cols]

    features = df[feature_cols].values.astype('float32')
    labels = df[label_cols].values.astype('float32')

    x_windows = []
    y_targets =[]
    
    # 2. Replicate keras.utils.timeseries_dataset_from_array
    for i in range(num_samples):
        # Grab 10 rows for the window
        window = features[i : i + sequence_length]
        # The target is the label at the END of the window
        target = labels[i + sequence_length - 1]
        
        x_windows.append(window)
        y_targets.append(target)
        
    x_windows = np.array(x_windows)
    y_targets = np.array(y_targets)

    print(f"Generated X shape: {x_windows.shape} -> Will format as ({num_samples}, {sequence_length}, 1, {len(feature_cols)})")
    print(f"Generated Y shape: {y_targets.shape}")

    # 3. Write to Arduino .h file
    with open(output_h_file, 'w') as f:
        f.write("#ifndef IMU_DATASET_H\n")
        f.write("#define IMU_DATASET_H\n\n")
        
        f.write("// Automatically generated 1D CNN Windowed Dataset\n")
        f.write(f"const int NUM_SAMPLES = {num_samples};\n")
        f.write(f"const int SEQUENCE_LENGTH = {sequence_length};\n")
        f.write(f"const int NUM_FEATURES = {len(feature_cols)};\n")
        f.write(f"const int NUM_OUTPUTS = {len(label_cols)};\n\n")
        
        # X_data formatting to match TF expand_dims(x, axis=2) ->[Batch][Seq][1][Features]
        f.write("// Shape: [NUM_SAMPLES][SEQUENCE_LENGTH][1][NUM_FEATURES]\n")
        f.write("const float X_data[NUM_SAMPLES][SEQUENCE_LENGTH][1][NUM_FEATURES] = {\n")
        for i, sample in enumerate(x_windows):
            f.write("    { // Sample " + str(i) + "\n")
            for j, step in enumerate(sample):
                step_str = ", ".join([f"{val:.6f}" for val in step])
                # Double braces to handle the extra '1' dimension (Width)
                f.write(f"        {{{{{step_str}}}}}")
                if j < sequence_length - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("    }")
            if i < num_samples - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")
        
        # Y_data formatting (Regression targets)
        f.write("// Shape: [NUM_SAMPLES][NUM_OUTPUTS]\n")
        f.write("// Columns: Euler_x, Euler_y, Euler_z, Quat_0, Quat_1, Quat_2, Quat_3\n")
        f.write("const float Y_data[NUM_SAMPLES][NUM_OUTPUTS] = {\n")
        for i, target in enumerate(y_targets):
            target_str = ", ".join([f"{val:.6f}" for val in target])
            f.write(f"    {{{target_str}}}")
            if i < num_samples - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")
        
        f.write("#endif // IMU_DATASET_H\n")
        
    print(f"✅ Successfully created {output_h_file}")

if __name__ == "__main__":
    # Point this to your actual CSV file
    generate_c_header('dataset/IMU_Data_1.csv', num_samples=5)