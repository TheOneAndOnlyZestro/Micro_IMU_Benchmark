import os
import json
import tensorflow as tf

TFLITE_DIR = 'tflite_models'
OUTPUT_JSON = 'tflite_structures.json'

def get_tflite_summary(model_path):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    ops = interpreter._get_ops_details()
    tensors = interpreter.get_tensor_details()
    
    lines = []
    lines.append(f"TFLite Model: {os.path.basename(model_path)}")
    lines.append(f"Total Tensors: {len(tensors)} | Total Operations: {len(ops)}")
    lines.append("=" * 60)
    
    # Extract Inputs
    lines.append("📥 INPUT TENSORS:")
    for idx in interpreter.get_input_details():
        lines.append(f"  [{idx['index']}] {idx['name']} : {idx['shape']} ({idx['dtype'].__name__})")
        
    # Extract Outputs
    lines.append("\n📤 OUTPUT TENSORS:")
    for idx in interpreter.get_output_details():
        lines.append(f"  [{idx['index']}] {idx['name']} : {idx['shape']} ({idx['dtype'].__name__})")
        
    # Extract Operation Graph
    lines.append("\n⚙️ EXECUTION GRAPH (Operations):")
    lines.append(f"{'Idx':<4} | {'Operation':<20} | {'Inputs':<15} -> {'Outputs'}")
    lines.append("-" * 60)
    
    for i, op in enumerate(ops):
        op_name = op['op_name']
        inputs = str(op['inputs'])
        outputs = str(op['outputs'])
        lines.append(f"{i:<4d} | {op_name:<20} | {inputs:<15} -> {outputs}")
        
    return "\n".join(lines)

def extract_tflite_structures():
    tflite_files = [f for f in os.listdir(TFLITE_DIR) if f.endswith('.tflite')]
    structures = {}

    print(f"🔍 Extracting structures for {len(tflite_files)} TFLite models...")

    for filename in tflite_files:
        filepath = os.path.join(TFLITE_DIR, filename)
        try:
            summary_str = get_tflite_summary(filepath)
            structures[filename] = summary_str
            print(f"  ✅ Extracted: {filename}")
        except Exception as e:
            print(f"  ❌ Failed to extract {filename}: {e}")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(structures, f, indent=4)
        
    print(f"\n🎉 Successfully saved structures to {OUTPUT_JSON}")

if __name__ == "__main__":
    extract_tflite_structures()