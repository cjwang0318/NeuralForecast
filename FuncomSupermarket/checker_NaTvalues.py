import pandas as pd


def check_for_NaT(df, date_columns):
    """
    Check specified date columns for NaT values and print the indices of such values.

    Parameters:
    df (pd.DataFrame): DataFrame to check.
    date_columns (list): List of column names to check for NaT values.

    Raises:
    ValueError: If any NaT values are found in the specified columns.
    """
    for col in date_columns:
        nat_indices = df[df[col].isna()].index
        if not nat_indices.empty:
            print(
                f"Column '{col}' contains NaT values at indices: {nat_indices.tolist()}")
            raise ValueError(
                f"Column '{col}' contains NaT values. Please clean your data.")


if __name__ == "__main__":
    # ds使用連續自然數
    filePath = "./dataset/2023_01_2024_06訂單彙整分店商品週期小資料_經國店.csv"
    print(filePath)
    df = pd.read_csv(filePath,
                     header=None, skiprows=1, names=["unique_id", "ds", "y"])
    print(df)
    # 檢查ds這個column是不是有包含NaTvalues
    try:
        check_for_NaT(df, ['ds'])
    except ValueError as e:
        print(e)
