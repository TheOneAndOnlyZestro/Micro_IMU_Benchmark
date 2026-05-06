import os
import json
import tensorflow as tf

# Import Keras model creation functions
from model import create_dense_n_model, create_conv_n_model, create_lstm_n_model

MODEL_DIR = 'model_weights'
OUTPUT_JSON = 'keras_structures.json'

def extract_keras_structures():
    if not os.path.exists(MODEL_DIR):
        print(f"⚠️ Directory {MODEL_DIR} not found.")
        return

    keras_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
    structures = {}

    print(f"🔍 Extracting structures for {len(keras_files)} Keras models...")

    for filename in keras_files:
        base_name = filename.replace('.keras', '')
        parts = base_name.split('_')
        
        arch = parts[0]
        n = int(parts[1])
        seq_length = int(parts[2])
        batch_norm = True if 'ON' in parts[3] else False

        try:
            if arch == 'dense':
                model = create_dense_n_model(n+1, batch_size=1)(sequence_length=seq_length, batch_normalization=batch_norm)
            elif arch == 'conv':
                model = create_conv_n_model(n+1, batch_size=1)(sequence_length=seq_length, batch_normalization=batch_norm)
            elif arch == 'lstm':
                model = create_lstm_n_model(n+1, batch_size=1)(sequence_length=seq_length, batch_normalization=batch_norm)
            else:
                continue
            
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            
            structures[base_name] = {
                "params": model.count_params(),
                "summary": "\n".join(summary_lines)
            }
            print(f"  ✅ Extracted: {base_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to extract {base_name}: {e}")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(structures, f, indent=4)
        
    print(f"\n🎉 Saved Keras structures to {OUTPUT_JSON}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU
    extract_keras_structures()