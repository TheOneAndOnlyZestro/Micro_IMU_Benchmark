import os
# # DISABLE GPU: This must be the first thing in your script
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence logs
# os.environ["TF_MLIR_OPTIMIZE_GML_TO_REG"] = "0" # Disable specific bugged optimizer

import tensorflow as tf

# DOUBLE DISABLE: Tell TF explicitly to hide the GPU
try:
    tf.config.set_visible_devices([], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    print(f"GPUs visible after disable: {len(logical_devices)}")
except:
    pass

# --- Your Conversion Logic ---
modeldir = 'model_weights'
filename = 'conv_lr0.005_bs256.keras' # or your LSTM model name
model_path = os.path.join(modeldir, filename)

print("Loading model...")
model = tf.keras.models.load_model(model_path)

print("Starting Conversion (on CPU)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# For LSTMs, these two lines are usually required:
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS 
]
converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()

# --- Save Logic ---
output_path = filename.replace('.keras', '.tflite')
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"Successfully saved to {output_path}")