import os
import pandas as pd
import numpy as np
import tensorflow as tf

# --- Configuration ---
MODEL_PATH = 'model_weights/conv_lr0.0001_bs32.keras'
CSV_PATH = 'dataset/IMU_Data_5.csv'
SEQ_LEN = 10
NUM_PREDICTIONS = 10  # How many windows to test

def run_test():
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"Loading data from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Separate Features and Labels
    label_cols = ["Euler_x", "Euler_y", "Euler_z", "Quat_0", "Quat_1", "Quat_2", "Quat_3"]
    feature_cols = [col for col in df.columns if col not in label_cols]
    
    # Shape: (Total_Rows, 9)
    features = df[feature_cols].values.astype('float32')
    # Shape: (Total_Rows, 7)
    labels = df[label_cols].values.astype('float32')
    
    # 2. Add the dummy dimension for your Conv2D model architecture
    # Shape becomes: (Total_Rows, 1, 9)
    features = np.expand_dims(features, axis=1)
    
    # 3. Create the input sequences (windows)
    X_test = []
    Y_true = []
    
    for i in range(NUM_PREDICTIONS):
        # Extract a window of 10 rows
        window = features[i : i + SEQ_LEN]
        X_test.append(window)
        
        # The target label is the label at the last timestep of the window
        target_label = labels[i + SEQ_LEN - 1]
        Y_true.append(target_label)
        
    # Convert lists to NumPy arrays
    X_test = np.array(X_test)  # Shape: (10, 10, 1, 9)
    Y_true = np.array(Y_true)  # Shape: (10, 7)
    
    print(f"Running prediction on input shape: {X_test.shape}...\n")
    
    # 4. Run Inference
    predictions = model.predict(X_test, verbose=0)
    
    # 5. Print Results & Calculate MSE
    total_mse = 0.0
    
    for i in range(NUM_PREDICTIONS):
        pred = predictions[i]
        true = Y_true[i]
        
        # Calculate MSE for this specific prediction (matches your C++ logic)
        mse = np.mean(np.square(pred - true))
        total_mse += mse
        
        print(f"--- Sample {i+1} ---")
        print(f"  Predicted Euler: {pred[0]:.4f}, {pred[1]:.4f}, {pred[2]:.4f}")
        print(f"     Actual Euler: {true[0]:.4f}, {true[1]:.4f}, {true[2]:.4f}")
        print(f"  Predicted Quat:  {pred[3]:.4f}, {pred[4]:.4f}, {pred[5]:.4f}, {pred[6]:.4f}")
        print(f"     Actual Quat:  {true[3]:.4f}, {true[4]:.4f}, {true[5]:.4f}, {true[6]:.4f}")
        print(f"  Sample MSE:      {mse:.4f}\n")
        
    print(f"=====================================")
    print(f" Average MSE over {NUM_PREDICTIONS} samples: {total_mse / NUM_PREDICTIONS:.4f}")
    print(f"=====================================")

if __name__ == "__main__":
    # Ensure CPU execution to avoid any GPU setup issues
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    run_test()