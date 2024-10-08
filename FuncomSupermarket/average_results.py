import pandas as pd


# Root directory and file name prefix
root_dir = './result/'
dir_prefix = '2023_01_2024_06訂單彙整分店商品週期小資料_線上_'
suffixes = ['光明店', '大埔店', '大墩店', '朝富店', '東山店',
            '東興店', '永春店', '河南店', '經國店', '黎明店']

file_prefix = 'evaluation_day'
days = [1, 2, 3]  # Example days

for day in days:
    file_suffixes = day
    print(f"{file_prefix}{file_suffixes}:")
    # Generate the full file paths
    file_paths = [
        f"{root_dir}{dir_prefix}{suffix}/{file_prefix}{file_suffixes}.csv"
        for suffix in suffixes
    ]
    # print(file_paths)

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

    # Sort the results by MAPE
    averages = averages.sort_values(by='MAPE')
    
    # Display the results
    print(averages.to_string(index=False)+"\n")
    
    # Save the result to a new CSV file
    averages.to_csv(
        f'./result/averages_{file_prefix}{file_suffixes}.csv', index=False)
