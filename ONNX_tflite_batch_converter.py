import os
import subprocess
import pandas as pd
import numpy as np
import tensorflow as tf
import onnx2tf

# --- Configuration ---
ONNX_DIR = 'onnx_models'
TFLITE_DIR = 'tflite_models'
HEADER_DIR = 'tflite_headers'
DATA_DIR = 'dataset'
CALIBRATION_FILE = os.path.join(DATA_DIR, 'IMU_Data_1.csv')

for d in [TFLITE_DIR, HEADER_DIR]:
    os.makedirs(d, exist_ok=True)

def get_representative_data(expected_shape, seq_length, num_samples=100):
    """
    Dynamically formats the calibration dataset to match whatever 
    input shape onnx2tf decided to generate, using the specific sequence length.
    """
    df = pd.read_csv(CALIBRATION_FILE)
    label_cols = ["Euler_x", "Euler_y", "Euler_z", "Quat_0", "Quat_1", "Quat_2", "Quat_3"]
    feature_cols = [col for col in df.columns if col not in label_cols]
    
    features = df[feature_cols].values.astype('float32')

    def gen():
        for i in range(num_samples):
            window = features[i : i + seq_length] # Use dynamic sequence length
            if len(window) < seq_length: 
                break
            
            # Start with PyTorch's native shape: (1, Channels, Seq, Width)
            window_t = np.transpose(window)
            base_shape = np.reshape(window_t, (1, 9, seq_length, 1)).astype(np.float32)
            
            # If onnx2tf transposed the model to TF's native NHWC (1, seq, 1, 9)
            if expected_shape[-1] == 9:
                # Transpose from NCHW to NHWC
                base_shape = np.transpose(base_shape, (0, 2, 3, 1))
            
            yield [base_shape]
            
    return gen

def convert_to_header(tflite_path, header_path):
    try:
        command = f"xxd -i {tflite_path} > {header_path}"
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print(f"  [Error] xxd failed: {e}")

def run_conversion_pipeline():
    onnx_files = [f for f in os.listdir(ONNX_DIR) if f.endswith('.onnx')]
    onnx_files.sort()

    for filename in onnx_files:
        onnx_path = os.path.join(ONNX_DIR, filename)
        base_name = filename.replace('.onnx', '')
        
        # Extract sequence length from filename (Format: arch_layers_seq_bn)
        parts = base_name.split('_')
        try:
            seq_length = int(parts[2])
        except (IndexError, ValueError):
            print(f"  ⚠️ Skipping {filename}: Could not parse sequence length.")
            continue
            
        temp_saved_model_dir = f"temp_tf_{base_name}"
        
        print(f"\n🚀 Processing ONNX Model: {filename} (Seq Len: {seq_length})")
        
        try:
            # 1. Convert ONNX to TF SavedModel
            onnx2tf.convert(
                input_onnx_file_path=onnx_path,
                output_folder_path=temp_saved_model_dir,
                copy_onnx_input_output_names_to_tflite=True,
                non_verbose=True
            )
            
            # 2. Dynamically check the input shape of the TF model
            tf_model = tf.saved_model.load(temp_saved_model_dir)
            infer = tf_model.signatures["serving_default"]
            expected_input_shape = list(infer.inputs[0].shape)
            print(f"  -> TF Model Expects Input Shape: {expected_input_shape}")

        except Exception as e:
            print(f"  ❌ Failed to convert ONNX to TF: {e}")
            continue

        # Removed 'float16' per request
        strategies = ['float32', 'hybrid', 'int8']

        for strat in strategies:
            print(f"  -> Converting Strategy: {strat}")
            
            converter = tf.lite.TFLiteConverter.from_saved_model(temp_saved_model_dir)
            
            # REQUIRED: Prevent errors on LSTMs during TFLite conversion
            converter._experimental_lower_tensor_list_ops = False
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

            if strat == 'hybrid':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
            elif strat == 'int8':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                # Pass the dynamic seq_length into the calibrator generator
                converter.representative_dataset = get_representative_data(expected_input_shape, seq_length)
                
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

            try:
                tflite_model = converter.convert()
                
                # Prepend 'torch_' to easily identify converted models in dashboard
                tflite_name = f"torch_{base_name}_{strat}.tflite"
                tflite_path = os.path.join(TFLITE_DIR, tflite_name)
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)

                header_name = f"torch_{base_name}_{strat}.h"
                header_path = os.path.join(HEADER_DIR, header_name)
                convert_to_header(tflite_path, header_path)
                
                print(f"     ✅ Success: {tflite_name}")
            except Exception as e:
                print(f"     ⚠️ Strategy {strat} failed: {e}")

if __name__ == "__main__":
    # Ensure CPU is used for TF operations
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    run_conversion_pipeline()
    print("\n✅ Phase 2: TF/TFLite Conversion & Header Generation Complete.")