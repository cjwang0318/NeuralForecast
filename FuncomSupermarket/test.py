import pandas as pd
# Reading an Excel file
# ds使用連續自然數
filePath = "./dataset/2023.csv"
df = pd.read_csv(filePath,
                 header=None, skiprows=1, names=["unique_id", "ds", "y"])
print(df.head())
# Counting the frequency of each ID
id_counts = df['unique_id'].value_counts()
print("Frequency of each unique_id:")
print(id_counts)
id_counts.to_csv('id_counts.csv', index=True)
