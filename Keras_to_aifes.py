import sys
import tensorflow as tf
from tensorflow import keras
from aifes.tools import create_flatbuffer_f32, print_flatbuffer_c_style

# -------------------------------------------------------------------------
# 1. Load the pre-trained Keras Model
# -------------------------------------------------------------------------
model_path = './model_weights/conv_lr0.0001_bs32.keras'
print(f"Loading model: {model_path}")
model = keras.models.load_model(model_path)

model.summary()

# -------------------------------------------------------------------------
# 2. Extract Weights and Create the AIfES F32 Flatbuffer
# -------------------------------------------------------------------------
# Keras stores weights as a list of Numpy arrays. 
# Extract them all sequentially.
weights = model.get_weights()

# Pack all the Numpy float arrays into a single, contiguous byte buffer 
# that the AIfES C-backend can digest directly.
flatbuffer_f32 = create_flatbuffer_f32(weights)

# -------------------------------------------------------------------------
# 3. Export the Flatbuffer to a C-Header (.h) File
# -------------------------------------------------------------------------
header_filename = "aifes_conv_model_weights.h"
print(f"\nGenerating C-header file: {header_filename}...")

# We temporarily redirect Python's standard output (stdout) to our .h file 
# so that the AIfES print function writes directly into it instead of the console.
original_stdout = sys.stdout

with open(header_filename, "w") as f:
    sys.stdout = f
    
    # Write the C-header guards
    print("#ifndef AIFES_CONV_MODEL_WEIGHTS_H")
    print("#define AIFES_CONV_MODEL_WEIGHTS_H\n")
    print("#include <stdint.h>\n")
    
    # Let the AIfES tool print the actual C-array definition.
    # It automatically handles the variable name and formatting!
    print_flatbuffer_c_style(flatbuffer_f32, elements_per_line=8)
    
    # Close the C-header guard
    print("\n#endif // AIFES_CONV_MODEL_WEIGHTS_H")

# VERY IMPORTANT: Restore standard output so normal print() works again
sys.stdout = original_stdout

print(f"✅ Successfully extracted weights and generated: {header_filename}!")