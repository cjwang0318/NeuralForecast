import pandas as pd

# ds使用連續自然數
filePath = "./dataset/2023.csv"
df = pd.read_csv(filePath,
                 header=None, skiprows=1, names=["unique_id", "ds", "y"])

# Counting the frequency of each ID
id_counts = df['unique_id'].value_counts()

print("Frequency of each ID:")
print(id_counts)

# Define the threshold，最小值要大於最佳化參數搜尋的的最大值，例如'input_size': tune.choice([5, 7, 14])就要大於14
threshold = 14

# Filter IDs based on the threshold
ids_to_keep = id_counts[id_counts >= threshold].index

# Filter the original DataFrame to keep only the desired IDs
filtered_df = df[df['unique_id'].isin(ids_to_keep)]

print("Filtered DataFrame:")
print(filtered_df)

# Output the filtered DataFrame to a CSV file
filtered_df.to_csv('filtered_data.csv', index=False)
