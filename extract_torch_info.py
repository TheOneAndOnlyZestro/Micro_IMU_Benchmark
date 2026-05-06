import os
import json
import torch

# Import PyTorch model creation classes
from torch_model import NDenseModelModifiable, NConvModelModifiable, NLSTMModelModifiable

MODEL_DIR = 'torch_model_weights'
OUTPUT_JSON = 'torch_structures.json'

def count_torch_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract_torch_structures():
    if not os.path.exists(MODEL_DIR):
        print(f"⚠️ Directory {MODEL_DIR} not found.")
        return

    pt_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
    structures = {}

    print(f"🔍 Extracting structures for {len(pt_files)} PyTorch models...")

    for filename in pt_files:
        base_name = filename.replace('.pt', '')
        
        # We prepend 'torch_' to the dictionary key so the Dashboard maps it correctly!
        dashboard_key = f"torch_{base_name}" 
        
        parts = base_name.split('_')
        arch = parts[0]
        n = int(parts[1])
        seq_length = int(parts[2])
        batch_norm = True if 'ON' in parts[3] else False

        try:
            if arch == 'dense': 
                model = NDenseModelModifiable(n, seq_length, batch_norm)
            elif arch == 'conv': 
                model = NConvModelModifiable(n, seq_length, batch_norm)
            elif arch == 'lstm': 
                model = NLSTMModelModifiable(n, seq_length, batch_norm)
            else:
                continue

            structures[dashboard_key] = {
                "params": count_torch_params(model),
                "summary": str(model) # PyTorch's native string representation of the architecture
            }
            print(f"  ✅ Extracted: {dashboard_key}")
            
        except Exception as e:
            print(f"  ❌ Failed to extract {base_name}: {e}")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(structures, f, indent=4)
        
    print(f"\n🎉 Saved PyTorch structures to {OUTPUT_JSON}")

if __name__ == "__main__":
    extract_torch_structures()