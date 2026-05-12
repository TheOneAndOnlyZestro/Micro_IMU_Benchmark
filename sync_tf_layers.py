import os
import glob
import pandas as pd
import json

CSV_FILES = glob.glob('benchmarks/*.csv')
KERAS_JSON = 'keras_structures.json'
TFLITE_JSON = 'tflite_structures.json'

def shift_tf_layer_name(old_name):
    """
    Shifts the layer number of TF models by +1.
    Ignores PyTorch models ('torch_...').
    Works for both full filenames (.tflite) and base names.
    """
    if old_name.startswith('torch_'):
        return old_name  # Leave PyTorch alone
        
    parts = old_name.split('_')
    
    # Ensure it matches the expected pattern (e.g. conv_0_...)
    if len(parts) >= 2 and parts[1].isdigit():
        parts[1] = str(int(parts[1]) + 1)
        return '_'.join(parts)
        
    return old_name

def sync_csv_files():
    if not CSV_FILES:
        print("⚠️ No CSV files found in 'benchmarks/' folder.")
        return

    print(f"🔄 Synchronizing {len(CSV_FILES)} CSV reports...")
    for file in CSV_FILES:
        df = pd.read_csv(file)
        
        if 'Model' in df.columns:
            # Apply the name shift to the Model column
            df['Model'] = df['Model'].apply(shift_tf_layer_name)
            df.to_csv(file, index=False)
            print(f"  ✅ Updated: {file}")

def sync_json_file(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found. Skipping.")
        return

    print(f"🔄 Synchronizing keys in {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)

    new_data = {}
    for old_key, value in data.items():
        new_key = shift_tf_layer_name(old_key)
        new_data[new_key] = value

    with open(filepath, 'w') as f:
        json.dump(new_data, f, indent=4)
        
    print(f"  ✅ Updated: {filepath}")

if __name__ == '__main__':
    print("="*50)
    print("Starting Keras (TF) Layer +1 Synchronization")
    print("="*50)
    sync_json_file(KERAS_JSON)
    sync_json_file(TFLITE_JSON)
    
    print("\n🎉 Synchronization complete! Your dashboard will now align PyTorch and Keras perfectly.")