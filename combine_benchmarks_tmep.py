import pandas as pd

# Load the CSV files
df1 = pd.read_csv('./benchmarks/benchmark_results_1778104302.csv')
df2 = pd.read_csv('./benchmarks/benchmark_results_1778112503.csv')

# Concat them (append df2 below df1)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save to a new file
combined_df.to_csv('./benchmarks/combined.csv', index=False)
