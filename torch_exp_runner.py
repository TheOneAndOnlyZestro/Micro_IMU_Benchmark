import os
import torch
import time
import pandas as pd
from torch.utils.data import DataLoader

# Import the new Modifiable architectures
from torch_dataset import IMUDataset
from torch_model import NDenseModelModifiable, NConvModelModifiable, NLSTMModelModifiable

# --- CONFIGURATION ---
EXPERIMENT_CONFIG = {
    'architectures': ['lstm'],
    'n_layers': [1, 2, 3, 4, 5, 6, 7, 8, 9],           # Depth variations
    'sequence_lengths': [5, 10, 15, 20],   # Temporal window sizes
    'batch_norms': [False, True],          # Batch Normalization OFF/ON
    
    # Standard Hyperparameters
    'learning_rate': 1e-4,
    'batch_size': 512,
    'epochs': 10,
}

# File paths
IMU_TRAIN_FILES = ["./dataset/IMU_Data_1.csv", "./dataset/IMU_Data_2.csv"]
IMU_TEST_FILES = ["./dataset/IMU_Data_5.csv"]

# Output directories
RESULTS_DIR = './torch_experiment_results'
MODELS_DIR = './torch_model_weights'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Define the device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def get_model(arch, n, seq_length, bn):
    """Factory function to dynamically create N-layer models."""
    if arch == 'dense':
        return NDenseModelModifiable(n=n, seq_length=seq_length, batch_norm=bn)
    elif arch == 'conv':
        return NConvModelModifiable(n=n, seq_length=seq_length, batch_norm=bn)
    elif arch == 'lstm':
        return NLSTMModelModifiable(n=n, seq_length=seq_length, batch_norm=bn)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
def run_experiment(arch, n, seq_length, bn, lr, batch_size):
    """Run a single experiment configuration."""
    
    # -------------------------------------------------------------
    # CRITICAL: Enforce Naming Convention for Dashboard Parsing!
    # Format: {arch}_{layers}_{seq}_{bn}.pt
    # -------------------------------------------------------------
    bn_str = "ON" if bn else "OFF"
    model_name = f"{arch}_{n}_{seq_length}_{bn_str}"
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    
    print(f"\n🚀 Training Model: {model_name}")
    start_time = time.time()
    
    try:
        # Create and map model to device
        model = get_model(arch, n, seq_length, bn)
        model = model.to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Note: We pass the dynamic seq_length to the dataset!
        imu_dataset = IMUDataset(IMU_TRAIN_FILES, sequence_length=seq_length)
        imu_data = DataLoader(dataset=imu_dataset, batch_size=batch_size, shuffle=True)
        
        loss_fn = torch.nn.MSELoss()
        
        # Train model
        for epoch in range(EXPERIMENT_CONFIG.get('epochs')):
            model.train()
            running_loss = 0.0 
            
            for batch, (x, y) in enumerate(imu_data):
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() 
                
            avg_train_loss = running_loss / len(imu_data)
            print(f"  [EPOCH {epoch+1}/{EXPERIMENT_CONFIG.get('epochs')}] Avg Train Loss: {avg_train_loss:.4f}")
    
        # Evaluate on test set
        model.eval()
        imu_dataset_test = IMUDataset(IMU_TEST_FILES, sequence_length=seq_length)
        imu_data_test = DataLoader(dataset=imu_dataset_test, batch_size=batch_size, shuffle=False)

        test_loss = 0.0

        with torch.no_grad():
            for x, y in imu_data_test:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                test_loss += loss_fn(pred, y).item()

        test_loss /= len(imu_data_test)
        print(f"  🎯 Final test Avg Loss: {test_loss:.4f}")

        # Save the model dict
        torch.save(model.state_dict(), model_path)
        
        elapsed_time = time.time() - start_time
        
        result = {
            'Model_Name': model_name,
            'Arch': arch,
            'Layers': n,
            'Seq_Length': seq_length,
            'BatchNorm': bn_str,
            'final_loss': test_loss,
            'training_time_seconds': elapsed_time
        }
        
        print(f"  ✅ Saved to: {model_path}")
        return result
        
    except Exception as e:
        print(f"  ❌ Failed: {model_name} - Error: {str(e)}")
        return None

def run_all_experiments():
    """Iterate over all dynamic combinations."""
    results = []
    
    lr = EXPERIMENT_CONFIG['learning_rate']
    bs = EXPERIMENT_CONFIG['batch_size']
    
    for arch in EXPERIMENT_CONFIG['architectures']:
        for n in EXPERIMENT_CONFIG['n_layers']:
            for seq_length in EXPERIMENT_CONFIG['sequence_lengths']:
                for bn in EXPERIMENT_CONFIG['batch_norms']:
                    
                    result = run_experiment(arch, n, seq_length, bn, lr, bs)
                    if result:
                        results.append(result)
                        
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
    
    results = run_all_experiments()
    
    if results:
        save_results_to_csv(results)
    else:
        print("\nNo successful experiments completed.")