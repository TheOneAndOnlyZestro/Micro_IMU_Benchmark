import os
import torch

from torch_model import NDenseModel, NConvModel, NLSTMModel, NConvLSTMModel

MODEL_DIR = 'torch_model_weights'
ONNX_DIR = 'onnx_models'
os.makedirs(ONNX_DIR, exist_ok=True)

def run_dense_only(model_id: str):
    params = model_id.split('_')
    n_layers = int(params[2])
    start_units = int(params[3])
    end_units = int(params[4])
    sq = int(params[5])
    bn = params[6].split('.')[0] == 'True'

    return NDenseModel(n_layers=n_layers, start_units=start_units, 
            end_units=end_units,  seq_length=sq, batch_norm=bn), sq
    
    
def run_conv_only(model_id: str):
    params = model_id.split('_')
    n_layers = int(params[2])
    base_filters = int(params[3])
    max_filters = int(params[4])
    sq = int(params[5])
    bn = params[6].split('.')[0] == 'True'

    return NConvModel(n_layers=n_layers, base_filters=base_filters, 
            max_filters=max_filters,  seq_length=sq, batch_norm=bn), sq

def run_lstm_only(model_id: str):
    params = model_id.split('_')
    n_layers = int(params[2])
    lstm_units = int(params[3])
    dense_head_units = [int(param) for param in params[4 : -2]]
    sq = int(params[-2])
    bn = params[-1].split('.')[0] == 'True'
    
    return NLSTMModel(n_layers=n_layers, lstm_units=lstm_units, 
    dense_head_units=dense_head_units,  seq_length=sq, batch_norm=bn) , sq

def run_conv_lstm(model_id: str):
    params = model_id.split('_')
    n_conv_layers = int(params)[2]
    n_lstm_layers = int(params)[3]
    lstm_units = int(params)[4]
    dense_head_units = [int(param) for param in params[5 : -2]]
    sq = int(params[-2])
    bn = params[-1].split('.')[0] == 'True'

    return NConvLSTMModel(n_conv_layers=n_conv_layers, n_lstm_layers=n_lstm_layers, lstm_units=lstm_units,
    dense_head_units=dense_head_units,  seq_length=sq, batch_norm=bn), sq

model_handle = {
    'dense_only': run_dense_only,
    'conv_only': run_conv_only,
    'lstm_only' : run_lstm_only,
    'conv_lstm' : run_conv_lstm
}

def export_all_to_onnx():
    pt_files = [f for f in os.listdir(MODEL_DIR)]

    for filename in pt_files:
        model_path = os.path.join(MODEL_DIR, filename)
        base_name = filename.replace('.pt', '')
        
        parts = '_'.join(base_name.split('_')[1 : ])
        
        print(f"\n🚀 Exporting PyTorch Model: {filename}")
        
        try:
            type = int('_'.join(parts.split('_')[0:2]))
            torch_model, sq = model_handle.get(type)(parts)
            
            torch_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            torch_model.eval()

            # Dynamic dummy input (1, Channels, Seq, Width)
            dummy_input = torch.randn(1, 9, sq, 1)

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