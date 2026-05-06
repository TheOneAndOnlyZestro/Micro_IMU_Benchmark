import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
CSV_FILE = './benchmarks/benchmark_results_1778068826.csv'
OUTPUT_DIR = 'graphs'

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

def process_data(file_path):
    print("Loading and cleaning data...")
    df = pd.read_csv(file_path)
    
    df['Inference_Time_ms'] = pd.to_numeric(df['Inference_Time_ms'], errors='coerce')
    df['Arena_Memory_Bytes'] = pd.to_numeric(df['Arena_Memory_Bytes'], errors='coerce')
    df['MSE'] = pd.to_numeric(df['MSE'], errors='coerce')
    
    df_agg = df.groupby(['Model', 'Seq_Length']).agg({
        'Inference_Time_ms': 'mean',
        'Arena_Memory_Bytes': 'max',
        'MSE': 'mean'
    }).reset_index()
    
    def extract_metadata(row):
        name = row['Model'].replace('.tflite', '')
        parts = name.split('_')
        return pd.Series({
            'Arch': parts[0].upper(),
            'Layers': int(parts[1]),
            'BN': parts[3],
            'Quantization': parts[4]
        })
        
    df_agg[['Arch', 'Layers', 'BN', 'Quantization']] = df_agg.apply(extract_metadata, axis=1)
    
    valid_data = df_agg.dropna(subset=['Inference_Time_ms']).copy()
    return valid_data

def plot_architectures(df):
    """Line chart: perfectly safe because X-axis explicitly separates the Layers."""
    g = sns.FacetGrid(df, col="Arch", sharey=False, height=5, aspect=1.2)
    g.map_dataframe(sns.lineplot, x="Layers", y="Inference_Time_ms", 
                    hue="Quantization", style="BN", markers=True, dashes=False, err_style=None)
    g.add_legend()
    g.set_axis_labels("Number of Layers (Depth)", "Inference Time (ms)")
    g.figure.suptitle("Scaling by Architecture & Depth", y=1.05, fontsize=16)
    
    plt.savefig(f"{OUTPUT_DIR}/1_Architecture_Scaling.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_sequence_lengths(df):
    """
    FIXED: We MUST filter to a single depth (e.g., Layers == 1) to make a fair 
    comparison between Conv, Dense, and LSTM.
    """
    # Force Apples-to-Apples comparison
    df_baseline = df[df['Layers'] == 1].copy()
    
    g = sns.catplot(
        data=df_baseline, kind="bar",
        x="Arch", y="Inference_Time_ms", hue="Quantization", col="Seq_Length",
        height=5, aspect=0.8, sharey=False
    )
    g.set_axis_labels("Architecture", "Inference Time (ms)")
    g.figure.suptitle("Apples-to-Apples: Base Models (1 Layer) across Seq Lengths", y=1.05, fontsize=16)
    
    plt.savefig(f"{OUTPUT_DIR}/2_Sequence_Lengths_FIXED.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_quantization_pareto(df):
    """Scatter Plot is safe because every individual model is a distinct dot."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, x="Arena_Memory_Bytes", y="Inference_Time_ms", 
        hue="Quantization", style="Arch", size="Layers", sizes=(50, 200), alpha=0.8
    )
    plt.title("Hardware Tradeoffs: int8 vs float32 (Memory vs. Speed)", fontsize=16)
    plt.xlabel("Arena Memory Used (Bytes)")
    plt.ylabel("Inference Time (ms)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Optional: Log scale helps if memory varies wildly
    # plt.xscale('log') 
    
    plt.savefig(f"{OUTPUT_DIR}/3_Quantization_Tradeoffs.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_batch_normalization(df):
    """
    FIXED: Batch Norm evaluation should also evaluate models at identical depths
    to ensure we aren't comparing deep BN=ON vs shallow BN=OFF.
    """
    # Filter to CONV models only, and let's pick a specific depth (e.g., 5 Layers)
    # where we have plenty of data for both ON and OFF
    df_bn = df[(df['Arch'] == 'CONV') & (df['Layers'] == 5)].copy()
    
    if df_bn.empty:
        print("Not enough data to plot Batch Norm fairly.")
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Impact of Batch Normalization (Locked at CONV, 5 Layers)", fontsize=16)
    
    sns.barplot(data=df_bn, x="BN", y="Inference_Time_ms", hue="Quantization", ax=axes[0])
    axes[0].set_title("Inference Time (ms)")
    axes[0].set_xlabel("Batch Normalization")
    
    sns.barplot(data=df_bn, x="BN", y="Arena_Memory_Bytes", hue="Quantization", ax=axes[1])
    axes[1].set_title("RAM Usage (Bytes)")
    axes[1].set_xlabel("Batch Normalization")
    
    plt.savefig(f"{OUTPUT_DIR}/4_BatchNorm_Impact_FIXED.png", bbox_inches='tight', dpi=300)
    plt.close()

def main():
    if not os.path.exists(CSV_FILE):
        print(f"File {CSV_FILE} not found!")
        return
        
    df = process_data(CSV_FILE)
    plot_architectures(df)
    plot_sequence_lengths(df)
    plot_quantization_pareto(df)
    plot_batch_normalization(df)
    print(f"\n✅ Fixed graphs saved successfully in the '{OUTPUT_DIR}' folder!")

if __name__ == '__main__':
    main()