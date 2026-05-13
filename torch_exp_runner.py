import os
import torch
import time
import pandas as pd
from torchinfo import summary
from torch.utils.data import DataLoader

# Import the new Modifiable architectures
from torch_dataset import IMUDataset
from torch_model import NDenseModel, NConvModel, NLSTMModel, NConvLSTMModel

BENCHMARK_SWEEP = {
    "dense_only": {
        "class": NDenseModel,
        "param_grid": [
            {"n_layers": n, "start_units": 128, "end_units": 16}
            for n in [1, 2, 3, 4, 5, 6, 8]
        ],
    },
    "conv_only": {
        "class": NConvModel,
        "param_grid": [
            {"n_layers": n, "base_filters": 16, "max_filters": 64}
            for n in [1, 2, 3, 4, 5, 6, 8]
        ],
    },
    "lstm_only": {
        "class": NLSTMModel,
        "param_grid": [
            {"n_layers": nl, "lstm_units": u, "dense_head_units": [18, 6]}
            for nl in [1, 2, 3]
            for u in [32,60]
        ],
    },
    "conv_lstm": {
        "class": NConvLSTMModel,
        "param_grid": [
            {
                "n_conv_layers": nc,
                "n_lstm_layers": nl,
                "lstm_units": u,
                "dense_head_units": [18, 6],
            }
            for nc in [2]
            for nl in [1, 2, 3]
            for u in [32, 60]
        ],
    },
}

# --- CONFIGURATION ---
EXPERIMENT_CONFIG = { 
    "sequence_lengths": [10, 20],
    "batch_norm_options": [False, True],
    'learning_rate': 1e-1,
    'batch_size': 1024,
    'epochs': 20,
}

# File paths
IMU_TRAIN_FILES = ["./dataset/IMU_Data_1.csv", "./dataset/IMU_Data_2.csv", "./dataset/IMU_Data_3.csv"]
IMU_TEST_FILES = ["./dataset/IMU_Data_5.csv"]

# Output directories
RESULTS_DIR = './torch_experiment_results'
MODELS_DIR = './torch_model_weights'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Define the device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def run_dense_only(model_info, model_module):
    n_layers = model_info.get('n_layers')
    start_units = model_info.get('start_units')
    end_units = model_info.get('end_units')

    model_id = f"{'dense_only'}_{n_layers}_{start_units}_{end_units}"
    for sq in EXPERIMENT_CONFIG.get('sequence_lengths'):
        for bn in EXPERIMENT_CONFIG.get('batch_norm_options'):
            yield (sq, bn, f"{model_id}_{sq}_{bn}", model_module(n_layers=n_layers,
                                        seq_length=sq, start_units=start_units, end_units=end_units,
                                        batch_norm=bn))

def run_conv_only(model_info, model_module):
    n_layers = model_info.get('n_layers')
    base_filters = model_info.get('base_filters')
    max_filters = model_info.get('max_filters')

    model_id = f"{'conv_only'}_{n_layers}_{base_filters}_{max_filters}"
    for sq in EXPERIMENT_CONFIG.get('sequence_lengths'):
        for bn in EXPERIMENT_CONFIG.get('batch_norm_options'):
            yield (sq, bn, f"{model_id}_{sq}_{bn}", model_module(n_layers=n_layers,
                                        seq_length=sq, base_filters=base_filters, max_filters=max_filters,
                                        batch_norm=bn))

def run_lstm_only(model_info, model_module):
    n_layers = model_info.get('n_layers')
    lstm_units = model_info.get('lstm_units')
    dense_head_units = model_info.get('dense_head_units')
    model_id = f"{'lstm_only'}_{n_layers}_{lstm_units}_{'_'.join(str(x) for x in dense_head_units)}"
    
    for sq in EXPERIMENT_CONFIG.get('sequence_lengths'):
        for bn in EXPERIMENT_CONFIG.get('batch_norm_options'):
            yield (sq, bn, f"{model_id}_{sq}_{bn}", model_module(n_layers=n_layers,
                                        seq_length=sq, lstm_units=lstm_units, 
                                        dense_head_units=dense_head_units,
                                        batch_norm=bn))

def run_conv_lstm(model_info, model_module):
    n_conv_layers = model_info.get('n_conv_layers')
    n_lstm_layers = model_info.get('n_lstm_layers')
    lstm_units = model_info.get('lstm_units')
    dense_head_units = model_info.get('dense_head_units')

    model_id = f"{'conv_lstm'}_{n_conv_layers}_{n_lstm_layers}_{lstm_units}_{'_'.join(str(x) for x in dense_head_units)}"
    
    for sq in EXPERIMENT_CONFIG.get('sequence_lengths'):
        for bn in EXPERIMENT_CONFIG.get('batch_norm_options'):
            yield (sq, bn, f"{model_id}_{sq}_{bn}", model_module(seq_length=sq, n_conv_layers= n_conv_layers,
                                                    n_lstm_layers=n_lstm_layers, lstm_units=lstm_units,
                                                    dense_head_units=dense_head_units))

model_handle = {
    'dense_only': run_dense_only,
    'conv_only': run_conv_only,
    'lstm_only' : run_lstm_only,
    'conv_lstm' : run_conv_lstm
}

def run_experiment(model, dataloader):
    """Run a single experiment configuration."""  

    # Create and map model to device
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=EXPERIMENT_CONFIG.get('learning_rate'))
    loss_fn = torch.nn.MSELoss()
    
    train_dataloader, test_dataloader = dataloader
    # Train model
    for epoch in range(EXPERIMENT_CONFIG.get('epochs')):
        model.train()
        running_loss = 0.0 
        
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() 
            
        avg_train_loss = running_loss / len(train_dataloader)
        print(f"  [EPOCH {epoch+1}/{EXPERIMENT_CONFIG.get('epochs')}] Avg Train Loss: {avg_train_loss:.4f}")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            batch_size = x.size(0)
            test_loss += loss_fn(pred, y).item() * batch_size  # un-average the batch loss
            total_samples += batch_size

    test_loss /= total_samples   # now equivalent to Keras
    print(f"  🎯 Final test Avg Loss: {test_loss:.4f}")

    return (test_loss, None, None)

def run_all_experiments(dataloaders):
    """Iterate over all dynamic combinations."""
    results = []
    
    
    for model_family, family_params in BENCHMARK_SWEEP.items():
        print(f"\n{'='*60}")
        print(f"Running experiments Family: {model_family}, ON Device {DEVICE}")
        print('=' * 60)
        model_module = family_params.get('class')
        model_permutations = family_params.get('param_grid')
        for model_info in model_permutations:
            models = model_handle.get(model_family)(model_info=model_info, model_module=model_module)  

            for sq, bn, model_id, current_model in models:
                start_time = time.time()
                loss, accuracy, history = run_experiment(current_model, dataloader=dataloaders.get(str(sq)))
                elapsed_time = time.time() - start_time

                # Save best model
                model_path = os.path.join(MODELS_DIR, f"torch_{model_id}.pt")
                torch.save(current_model.state_dict(), model_path)
                
                info_path = os.path.join(MODELS_DIR, f"torch_{model_id}.txt")
                with open(info_path, "w", encoding='utf-8') as f:
                    sample = 0
                    for x,y in dataloaders[str(EXPERIMENT_CONFIG.get('sequence_lengths')[0])][0]:
                        sample = x
                        break
                    f.write(str(summary(current_model, input_size=sample.shape, verbose=0)))

                result = {
                    'model_type': model_family,
                    'sequence_length' : sq,
                    'batch_normalization' : 'ON' if bn else 'OFF',
                    'final_loss': loss,
                    'training_time_seconds': elapsed_time,
                    'model_path': model_path
                }
                
                results.append(result)

                print(f"✓ Completed: {model_id}")
                print(result)

    return results

def save_results_to_csv(results):
    """Save all experiment results to CSV file."""
    df = pd.DataFrame(results)
    output_file = os.path.join(RESULTS_DIR, f"torch_training_summary_{int(time.time())}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("TORCH EXPERIMENT SUMMARY")
    print('=' * 60)
    
    print(df.to_string(index=False))
    print(f"\nTraining Results saved to: {output_file}")
    
    # Find best model
    best = df.loc[df['final_loss'].idxmin()]
    print("\n🏆 Best PyTorch Model:")
    print(best.to_string())

if __name__ == "__main__":
    print("Starting Automated PyTorch N-Modifiable Experiment Runner")
    print("=" * 60)
    
    dataloaders = {}
    for sq in EXPERIMENT_CONFIG.get('sequence_lengths'):
        train_dataset = IMUDataset(IMU_TRAIN_FILES, sequence_length=sq)
        train_data = DataLoader(dataset=train_dataset, batch_size=EXPERIMENT_CONFIG.get('batch_size'), shuffle=True)

        test_dataset = IMUDataset(IMU_TEST_FILES, sequence_length=sq)
        test_data = DataLoader(dataset=test_dataset, batch_size=EXPERIMENT_CONFIG.get('batch_size'), shuffle=False)
        
        dataloaders[str(sq)] = (train_data, test_data)
    
    results = run_all_experiments(dataloaders=dataloaders)
    
    if results:
        save_results_to_csv(results)
    else:
        print("\nNo successful experiments completed.")