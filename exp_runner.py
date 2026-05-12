import tensorflow as tf
import keras
import pandas as pd
import os
import time
from dataset import load_dataset_conv
from model import create_conv_lstm_model_generator, create_dense_model_generator, create_conv_model_generator, create_lstm_model_generator

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

BENCHMARK_SWEEP = {
    # Family 1: Dense-only depth sweep
    "dense_only": {
        "generator": create_dense_model_generator,
        "param_grid": [
            {"n_layers": n, "start_units": 128, "end_units": 16}
            for n in [1, 2, 3, 4, 5, 6, 8]
        ],
    },
    # Family 2: Conv2D-only depth sweep
    "conv_only": {
        "generator": create_conv_model_generator,
        "param_grid": [
            {"n_layers": n, "base_filters": 16, "max_filters": 64}
            for n in [1, 2, 3, 4, 5, 6, 8]
        ],
    },
    # Family 3: LSTM-only depth+width sweep
    # Note: keep sequence_length small here since unroll explodes graph size.
    "lstm_only": {
        "generator": create_lstm_model_generator,
        "param_grid": [
            {"n_layers": nl, "lstm_units": u, "dense_head_units": [18, 6]}
            for nl in [1, 2, 3]
            for u in [32, 60]
        ],
    },
    # Family 4: Conv + LSTM hybrid (closest to target)
    "conv_lstm": {
        "generator": create_conv_lstm_model_generator,
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
    "sequence_lengths" : [10,20],
    "batch_norm_options" : [False, True],
    'learning_rate': 1e-4,
    'batch_size': 512,
    'epochs': 10,
}


# File paths
IMU_TRAIN_FILES = ["./dataset/IMU_Data_1.csv", "./dataset/IMU_Data_2.csv", "./dataset/IMU_Data_3.csv"]
IMU_TEST_FILES = ["./dataset/IMU_Data_5.csv"]

# Output directories
RESULTS_DIR = './experiment_results'
MODELS_DIR = './model_weights'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def run_experiment(model, dataset):
    """Run a single experiment configuration."""
    optimizer = keras.optimizers.Adam(learning_rate=EXPERIMENT_CONFIG.get('learning_rate'))
    # Compile with specific learning rate
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    train_dataset, test_dataset = dataset
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EXPERIMENT_CONFIG.get('epochs'),
        batch_size=EXPERIMENT_CONFIG.get('batch_size'),
        verbose=1
    )

    loss, accuracy = model.evaluate(test_dataset, verbose=0)

    return (loss, accuracy, history)  

def run_dense_only(model_info, generator_func):
    n_layers = model_info.get('n_layers')
    start_units = model_info.get('start_units')
    end_units = model_info.get('end_units')

    model_id = f"{'dense_only'}_{n_layers}_{start_units}_{end_units}"
    current_model_generator = generator_func(n_layers=n_layers,
                                             start_units=start_units,
                                             end_units=end_units,
                                             batch_size=None)
    
    for sq in EXPERIMENT_CONFIG.get('sequence_lengths'):
        for bn in EXPERIMENT_CONFIG.get('batch_norm_options'):
            yield (sq, bn, model_id, current_model_generator(sequence_length=sq, batch_normalization= bn))

def run_conv_only(model_info, generator_func):
    n_layers = model_info.get('n_layers')
    base_filters = model_info.get('base_filters')
    max_filters = model_info.get('max_filters')

    model_id = f"{'conv_only'}_{n_layers}_{base_filters}_{max_filters}"
    current_model_generator = generator_func(n_layers=n_layers,
                                             base_filters=base_filters,
                                             max_filters=max_filters,
                                             batch_size=None)
    
    for sq in EXPERIMENT_CONFIG.get('sequence_lengths'):
        for bn in EXPERIMENT_CONFIG.get('batch_norm_options'):
            yield (sq, bn, model_id, current_model_generator(sequence_length=sq, batch_normalization= bn))

def run_lstm_only(model_info, generator_func):
    n_layers = model_info.get('n_layers')
    lstm_units = model_info.get('lstm_units')
    dense_head_units = model_info.get('dense_head_units')
    model_id = f"{'lstm_only'}_{n_layers}_{lstm_units}_{'_'.join(dense_head_units)}"
    current_model_generator = generator_func(n_layers=n_layers,
                                             lstm_units=lstm_units,
                                             dense_head_units=dense_head_units,
                                             batch_size=None)
    
    for sq in EXPERIMENT_CONFIG.get('sequence_lengths'):
        for bn in EXPERIMENT_CONFIG.get('batch_norm_options'):
            yield (sq, bn, model_id, current_model_generator(sequence_length=sq, batch_normalization= bn))

def run_conv_lstm(model_info, generator_func):
    n_conv_layers = model_info.get('n_conv_layers')
    n_lstm_layers = model_info.get('n_lstm_layers')
    lstm_units = model_info.get('lstm_units')
    dense_head_units = model_info.get('dense_head_units')

    model_id = f"{'conv_lstm'}_{n_conv_layers}_{n_lstm_layers}_{lstm_units}_{'_'.join(dense_head_units)}"
    current_model_generator = generator_func(n_conv_layers=n_conv_layers,
                                             n_lstm_layers=n_lstm_layers,
                                             lstm_units=lstm_units,
                                             dense_head_units=dense_head_units,
                                             batch_size=None)
    
    for sq in EXPERIMENT_CONFIG.get('sequence_lengths'):
        for bn in EXPERIMENT_CONFIG.get('batch_norm_options'):
            yield (sq, bn, model_id, current_model_generator(sequence_length=sq, batch_normalization= bn))

model_handle = {
    'dense_only': run_dense_only,
    'conv_only': run_conv_only,
    'lstm_only' : run_lstm_only,
    'conv_lstm' : run_conv_lstm
}
def run_all_experiments(datasets):
    """Run all model and hyperparameter combinations."""
    results = []
    
    for model_family, family_params in BENCHMARK_SWEEP.items():
        print(f"\n{'='*60}")
        print(f"Running experiments Family: {model_family}, ON Device {tf.config.list_physical_devices('GPU')}")
        print('=' * 60)
        
        generator_func = family_params.get('generator')
        model_permutations = family_params.get('param_grid')
        for model_info in model_permutations:
            models = model_handle.get(model_family)(model_info=model_info, generator_func=generator_func)
            
            for sq, bn, model_id, current_model in models:
                start_time = time.time()
                loss, accuracy, history = run_experiment(current_model, dataset=datasets.get(str(sq)))
                elapsed_time = time.time() - start_time

                # Save best model
                model_path = os.path.join(MODELS_DIR, f"{model_id}.keras")
                current_model.save(model_path)
                
                info_path = os.path.join(MODELS_DIR, f"{model_id}.txt")
                with open(info_path, "w") as f:
                    current_model.summary(print_fn=lambda x: f.write(x + "\n"))

                result = {
                    'model_type': model_family,
                    'sequence_length' : sq,
                    'batch_normalization' : 'ON' if bn else 'OFF',
                    'final_loss': loss,
                    'final_accuracy': accuracy,
                    'epochs_trained': len(history.history['loss']),
                    'training_time_seconds': elapsed_time,
                    'model_path': model_path
                }
                
                results.append(result)

                print(f"✓ Completed: {model_id}")
                print(result)


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
    

    # Prefetch datasets with all sq_lengths needed
    train_dataset_seq_10 = load_dataset_conv(
            imu_file=IMU_TRAIN_FILES, 
            sequence_length=10,
            batch_size=EXPERIMENT_CONFIG.get('batch_size')
    )
    test_dataset_seq_10 = load_dataset_conv(
            imu_file=IMU_TEST_FILES, 
            sequence_length=10,
            batch_size=EXPERIMENT_CONFIG.get('batch_size')
    )

    train_dataset_seq_20 = load_dataset_conv(
            imu_file=IMU_TRAIN_FILES, 
            sequence_length=20,
            batch_size=EXPERIMENT_CONFIG.get('batch_size')
    )
    test_dataset_seq_20 = load_dataset_conv(
            imu_file=IMU_TRAIN_FILES, 
            sequence_length=20,
            batch_size=EXPERIMENT_CONFIG.get('batch_size')
    )

    datsets = {
        '10' : (train_dataset_seq_10, test_dataset_seq_10),
        '20' : (train_dataset_seq_20, test_dataset_seq_20)
    }

    results = run_all_experiments(datsets)
    
    if results:
        save_results_to_csv(results)
    else:
        print("\nNo successful experiments completed.")
