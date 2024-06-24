import pandas as pd

# Root directory and file name prefix
root_dir = './result/'
dir_prefix = '2023_01_2024_06訂單彙整分店商品週期小資料_'
suffixes = ['光明店', '大埔店']  # Example array

# Generate the full file paths
file_paths = [
    f"{root_dir}{dir_prefix}{suffix}/evaluation_day1.csv" for suffix in suffixes]

# Initialize an empty list to hold DataFrames
dfs = []

# Loop through the file paths and read the CSV files
for file_path in file_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs)

# Calculate the averages of MAPE and RMSE by model
averages = combined_df.groupby(
    'Model')[['MAPE', 'RMSE', 'Training_Time']].mean().reset_index()

# Round the results to the fourth decimal place
averages = averages.round({'MAPE': 4, 'RMSE': 4, 'Training_Time': 4})

# Display the results
print(averages)

# Save the result to a new CSV file
averages.to_csv('./result/averages_by_model.csv', index=False)
