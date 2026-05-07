import pandas as pd

# Load the CSV files
# df1 = pd.read_csv('./benchmarks/benchmark_results_1778104302.csv')
# df2 = pd.read_csv('./benchmarks/benchmark_results_1778112503.csv')

# # Concat them (append df2 below df1)
# combined_df = pd.concat([df1, df2], ignore_index=True)

# # Save to a new file
# combined_df.to_csv('./benchmarks/combined.csv', index=False)


df_combined = pd.read_csv('./benchmarks/combined.csv')

#remove all with torch_dense_*_*_ON

# 1. Create the conditions (wrapped in parentheses)
starts_with = df_combined['Model'].str.startswith('torch_dense')
contains_on = df_combined['Model'].str.contains('ON')

# 2. Combine them with the bitwise AND operator (&)
condition_to_remove = (starts_with) & (contains_on)

# 3. Use the bitwise NOT operator (~) to keep everything EXCEPT those rows
df_new = df_combined[~condition_to_remove]

df_new.to_csv('./benchmarks/filtered.csv', index=False)