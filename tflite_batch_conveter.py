import os
import subprocess
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import your model creation functions (Ensure these use batch_size=1 as discussed!)
from model import create_dense_model_generator, create_conv_model_generator, create_lstm_model_generator, create_conv_lstm_model_generator

# --- Configuration ---
MODEL_DIR = 'model_weights'
DATA_DIR = 'dataset'
TFLITE_DIR = 'tflite_models'
HEADER_DIR = 'tflite_headers'

# Use one specific file for representative data to keep it consistent
CALIBRATION_FILE = os.path.join(DATA_DIR, 'IMU_Data_1.csv')

# Ensure directories exist
for d in [TFLITE_DIR, HEADER_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Data Preparation Logic ---

def get_representative_data(num_samples=100, sequence_length=10):
    """
    Prepares features for quantization calibration.
    Matches the (Batch, Seq, 1, Channels) shape requirement.
    """
    df = pd.read_csv(CALIBRATION_FILE)
    label_cols = ["Euler_x", "Euler_y", "Euler_z", "Quat_0", "Quat_1", "Quat_2", "Quat_3"]
    feature_cols = [col for col in df.columns if col not in label_cols]
    
    features = df[feature_cols].values.astype('float32')
    features = np.expand_dims(features, axis=1)

    def gen():
        for i in range(num_samples):
            window = features[i : i + sequence_length]
            if len(window) < sequence_length: 
                break
            yield [np.expand_dims(window, axis=0)]
            
    return gen

def run_dense_only(model_id: str):
    params = model_id.split('_')
    n_layers = int(params[2])
    start_units = int(params[3])
    end_units = int(params[4])
    sq = int(params[5])
    bn = params[6].split('.')[0] == 'True'

    return create_dense_model_generator(n_layers=n_layers, start_units=start_units, 
            end_units=end_units, batch_size=1)(sequence_length=sq, batch_normalization=bn), sq
    
    
def run_conv_only(model_id: str):
    params = model_id.split('_')
    n_layers = int(params[2])
    base_filters = int(params[3])
    max_filters = int(params[4])
    sq = int(params[5])
    bn = params[6].split('.')[0] == 'True'

    return create_conv_model_generator(n_layers=n_layers, base_filters=base_filters, 
            max_filters=max_filters, batch_size=1)(sequence_length=sq, batch_normalization=bn), sq

def run_lstm_only(model_id: str):
    params = model_id.split('_')
    n_layers = int(params[2])
    lstm_units = int(params[3])
    dense_head_units = [int(param) for param in params[4 : -2]]
    sq = int(params[-2])
    bn = params[-1].split('.')[0] == 'True'
    
    return create_lstm_model_generator(n_layers=n_layers, lstm_units=lstm_units, 
    dense_head_units=dense_head_units, batch_size=1)(sequence_length=sq, batch_normalization=bn) , sq

def run_conv_lstm(model_id: str):
    params = model_id.split('_')
    n_conv_layers = int(params)[2]
    n_lstm_layers = int(params)[3]
    lstm_units = int(params)[4]
    dense_head_units = [int(param) for param in params[5 : -2]]
    sq = int(params[-2])
    bn = params[-1].split('.')[0] == 'True'

    return create_conv_lstm_model_generator(n_conv_layers=n_conv_layers, n_lstm_layers=n_lstm_layers, lstm_units=lstm_units,
    dense_head_units=dense_head_units, batch_size=1)(sequence_length=sq, batch_normalization=bn), sq
model_handle = {
    'dense_only': run_dense_only,
    'conv_only': run_conv_only,
    'lstm_only' : run_lstm_only,
    'conv_lstm' : run_conv_lstm
}
# --- Conversion Logic ---
def convert_to_header(tflite_path, header_path):
    """System call to xxd to generate C++ headers."""
    try:
        command = f"xxd -i {tflite_path} > {header_path}"
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print(f"  [Error] xxd failed: {e}")

def run_benchmark_conversion():
    keras_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
    keras_files.sort()

    for filename in keras_files:
        model_path = os.path.join(MODEL_DIR, filename)
        base_name = filename.replace('.keras', '')
        
        print(f"\n🚀 Processing Sequence Model: {filename}")
        
        try:
            # 1. Load the dynamic model trained during the experiment
            trained_model = tf.keras.models.load_model(model_path)
            
            type = int('_'.join(filename.split('_')[0:2]))

            static_model, sq = model_handle.get(type)(filename)

            # 3. Transfer the weights
            static_model.set_weights(trained_model.get_weights())
            
        except Exception as e:
            print(f"  ❌ Failed to load/rebuild model: {e}")
            continue

        strategies = ['float32', 'int8']

        for strat in strategies:
            print(f"  -> Converting Strategy: {strat}")
            
            # Use the newly created STATIC model for conversion
            converter = tf.lite.TFLiteConverter.from_keras_model(static_model)
            
            # MANDATORY for Unrolled LSTMs on microcontrollers
            converter._experimental_lower_tensor_list_ops = False

            # Base supported ops (no SELECT_TF_OPS allowed for pure TFLM!)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

            if strat == 'int8':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = get_representative_data(num_samples=100, sequence_length=sq)
                
                # Strict Full Integer Quantization
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                
                # These ensure that the ESP32 input->type is kTfLiteInt8
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

            try:
                tflite_model = converter.convert()
                
                # Save .tflite
                tflite_name = f"{base_name}_{strat}.tflite"
                tflite_path = os.path.join(TFLITE_DIR, tflite_name)
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)

                # Save .h via xxd
                header_name = f"{base_name}_{strat}.h"
                header_path = os.path.join(HEADER_DIR, header_name)
                convert_to_header(tflite_path, header_path)
                
                print(f"     ✅ Success: {tflite_name}")
            except Exception as e:
                print(f"     ⚠️ Strategy {strat} failed: {e}")

if __name__ == "__main__":
    # Ensure we stay on CPU for conversion to avoid the M4/Metal bug
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    run_benchmark_conversion()
    print("\n✅ All conversions and header generation complete.")