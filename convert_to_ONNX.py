import os
import torch

from torch_model import NDenseModelModifiable, NConvModelModifiable, NLSTMModelModifiable

MODEL_DIR = 'torch_model_weights'
ONNX_DIR = 'onnx_models'
os.makedirs(ONNX_DIR, exist_ok=True)

def export_all_to_onnx():
    pt_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('lstm') and f.endswith('.pt')]

    for filename in pt_files:
        model_path = os.path.join(MODEL_DIR, filename)
        base_name = filename.replace('.pt', '')
        
        # e.g., "conv_7_10_ON"
        parts = base_name.split('_')
        arch = parts[0]
        n = int(parts[1])
        seq_length = int(parts[2])
        batch_norm = True if 'ON' in parts[3] else False
        
        print(f"\n🚀 Exporting PyTorch Model: {filename}")
        
        try:
            if arch == 'dense':
                torch_model = NDenseModelModifiable(n, seq_length, batch_norm)
            elif arch == 'conv':
                torch_model = NConvModelModifiable(n, seq_length, batch_norm)
            elif arch == 'lstm':
                torch_model = NLSTMModelModifiable(n, seq_length, batch_norm)
            else:
                continue
            
            torch_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            torch_model.eval()

            # Dynamic dummy input (1, Channels, Seq, Width)
            dummy_input = torch.randn(1, 9, seq_length, 1)

            onnx_path = os.path.join(ONNX_DIR, f"{base_name}.onnx")
            torch.onnx.export(
                torch_model, dummy_input, onnx_path,
                input_names=['input'], output_names=['output'],
                opset_version=16,
                do_constant_folding=True
            )
            print(f"  ✅ Saved: {onnx_path}")

        except Exception as e:
            print(f"  ❌ Failed to export model: {e}")

if __name__ == "__main__":
    export_all_to_onnx()