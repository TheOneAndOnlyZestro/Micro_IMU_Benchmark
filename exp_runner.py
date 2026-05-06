import tensorflow as tf
import keras
import pandas as pd
import os
import time
from dataset import load_dataset_conv
from model import create_dense_n_model,create_conv_n_model, create_lstm_n_model, create_residual_conv_lstm_model


# --- CONFIGURATION ---
EXPERIMENT_CONFIG = {
    'models': {
        **{f"conv_{i}": create_conv_n_model(n= i + 1) for i in range(10)}, 
        **{f"dense_{i}": create_dense_n_model(n= i + 1) for i in range(10)},
        **{f"lstm_{i}": create_lstm_n_model(n= i + 1) for i in range(10)}
    },
    'learning_rates': [1e-4],
    'batch_sizes': [512],
    'epochs': 10,
    'sequence_length': [5, 10, 15, 20],
    'val_split': 0.2,
}

# File paths
IMU_TRAIN_FILES = ["./dataset/IMU_Data_1.csv", "./dataset/IMU_Data_2.csv"]
IMU_TEST_FILES = ["./dataset/IMU_Data_5.csv"]

# Output directories
RESULTS_DIR = './experiment_results'
MODELS_DIR = './model_weights'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def run_experiment(model_name, model_func, sequence_length, lr, batch_size, batch_normalization):
    """Run a single experiment configuration."""
    start_time = time.time()
    
    try:
        # Create model
        model = model_func(sequence_length, batch_normalization)
        
        # Compile with specific learning rate
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        
        train_dataset = load_dataset_conv(
            imu_file=IMU_TRAIN_FILES, 
            sequence_length=sequence_length,
            batch_size=EXPERIMENT_CONFIG['batch_sizes'][0]
        )
        
        test_dataset = load_dataset_conv(
            imu_file=IMU_TEST_FILES, 
            sequence_length=sequence_length,
            batch_size=EXPERIMENT_CONFIG['batch_sizes'][0]
        )
        
        # Early stopping callback
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=EXPERIMENT_CONFIG['epochs'],
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate on test set
        loss, accuracy = model.evaluate(test_dataset, verbose=0)
        
        # Save best model
        model_path = os.path.join(MODELS_DIR, f"{model_name}_{sequence_length}_{'ON' if batch_normalization else 'OFF'}.keras")
        model.save(model_path)
        
        elapsed_time = time.time() - start_time
        
        result = {
            'model_type': model_name,
            'learning_rate': lr,
            'batch_size': batch_size,
            'sequence_length' : sequence_length,
            'batch_normalization' : 'ON' if batch_normalization else 'OFF',
            'final_loss': loss,
            'final_accuracy': accuracy,
            'epochs_trained': len(history.history['loss']),
            'training_time_seconds': elapsed_time,
            'model_path': model_path
        }
        
        print(f"✓ Completed: {model_name} | SQ:{sequence_length} | LR:{lr:.0e} | BS:{batch_size} | BN:{'ON' if batch_normalization else 'OFF'}")
        return result
        
    except Exception as e:
        print(f"✗ Failed: {model_name} | SQ:{sequence_length} | LR:{lr} | BS:{batch_size} | BN:{'ON' if batch_normalization else 'OFF'} - Error: {str(e)}")
        return None


def run_all_experiments():
    """Run all model and hyperparameter combinations."""
    results = []
    
    for model_name, create_func in EXPERIMENT_CONFIG['models'].items():
        print(f"\n{'='*60}")
        print(f"Running experiments for: {model_name.upper()}, ON Device {tf.config.list_physical_devices('GPU')}")
        print('=' * 60)
        
        for lr in EXPERIMENT_CONFIG['learning_rates']:
            for bs in EXPERIMENT_CONFIG['batch_sizes']:
                for sq in EXPERIMENT_CONFIG['sequence_length']:

                    if model_name.startswith("conv"):
                        for bn in [False, True]:
                            result = run_experiment(model_name, create_func, sq ,lr, bs, bn)
                            if result:
                                results.append(result)
                    else:
                        result = run_experiment(model_name, create_func, sq ,lr, bs, False)
                        if result:
                            results.append(result)
    
    return results


def save_results_to_csv(results):
    """Save all experiment results to CSV file."""
    df = pd.DataFrame(results)
    output_file = os.path.join(RESULTS_DIR, f"experiment_results_{int(time.time())}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print('=' * 60)
    print(df[['model_type', 'learning_rate', 'batch_size', 'sequence_length', 'batch_normalization',
              'final_loss', 'final_accuracy']].to_string(index=False))
    print(f"\nResults saved to: {output_file}")
    
    # Find best model
    best = df.loc[df['final_loss'].idxmin()]
    print("\n🏆 Best Model:")
    print(best.to_string())


if __name__ == "__main__":
    print("Starting Automated ML Experiment Runner")
    print("=" * 60)
    #tf.config.set_visible_devices([], 'GPU')
    # Run all experiments
    results = run_all_experiments()
    
    if results:
        save_results_to_csv(results)
    else:
        print("\nNo successful experiments completed.")
