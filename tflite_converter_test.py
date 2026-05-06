import os
# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

def run_test():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    
    # 1. Create a minimal model
    print("\n--- Step 1: Creating Minimal Model ---")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print("Model created successfully.")

    # 2. Test Direct Conversion
    print("\n--- Step 2: Testing Direct Conversion (In-Memory) ---")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        print("SUCCESS: Direct conversion worked.")
    except Exception as e:
        print(f"FAILED: Direct conversion failed with error:\n{e}")

    # 3. Test .keras File Conversion
    print("\n--- Step 3: Testing .keras File Serialization ---")
    test_filename = "test_model_delete_me.keras"
    try:
        model.save(test_filename)
        reloaded_model = tf.keras.models.load_model(test_filename)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(reloaded_model)
        tflite_model = converter.convert()
        print("SUCCESS: .keras file conversion worked.")
    except Exception as e:
        print(f"FAILED: .keras file conversion failed with error:\n{e}")
    finally:
        if os.path.exists(test_filename):
            os.remove(test_filename)

if __name__ == "__main__":
    run_test()