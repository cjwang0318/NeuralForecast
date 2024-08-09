import config as cf
import json
from neuralforecast.core import NeuralForecast
from neuralforecast.models import PatchTST, LSTM
from neuralforecast.auto import AutoLSTM, AutoPatchTST, AutoNHITS, AutoAutoformer 
from neuralforecast.losses.pytorch import MQLoss, MAPE, HuberMQLoss, MAE
from neuralforecast.losses.numpy import mape, rmse
import time
import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# from ray import tune
# from ray.tune.search.hyperopt import HyperOptSearch

# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def insert_ds(df):
    uids = df["unique_id"].unique()  # Select ids
    df_return = pd.DataFrame()
    for id in uids:
        df_temp = df.query("unique_id in @id").reset_index(drop=True)
        start = 1
        df_temp.insert(2, "ds", range(start, start + df_temp.shape[0]))
        # print(Y_df.shape)
        # print(Y_df.head(20))
        df_return = pd.concat([df_return, df_temp], axis=0, ignore_index=True)
    df_return.drop("datetime", inplace=True, axis=1)
    # print(df_return.tail())
    return df_return


def model_training(df, horizon, model, n_windows, num_samples, checkpointsPath):
    # Utilizing Auto-models for parameters tuning by random grid search, and choose the best parameters by cross validation
    if model == "LSTM":
        models = [
            AutoLSTM(
                h=horizon, config=cf.config_lstm, loss=MQLoss(), num_samples=num_samples
            )
        ]
    elif model == "PatchTST":
        models = [
            AutoPatchTST(
                h=horizon,
                config=cf.config_patchtst,
                loss=MQLoss(),
                num_samples=num_samples,
            )
        ]
    elif model == "NHITS":
        models = [
            AutoNHITS(
                h=horizon,
                config=cf.config_nhits,
                loss=MQLoss(),
                num_samples=num_samples,
            )
        ]
    elif model == "Autoformer":
        models = [
            AutoAutoformer(
                h=horizon,
                config=cf.config_Autoformer,
                loss=MQLoss(),
                num_samples=num_samples,
            )
        ]
    else:
        print("NO model is selected...")
        # models = [
        #     AutoLSTM(h=horizon, config=cf.config_lstm,
        #              loss=MQLoss(), num_samples=num_samples),
        #     AutoPatchTST(h=horizon, config=config_patchtst,
        #                  loss=MQLoss(), num_samples=num_samples),
        #     AutoNHITS(h=horizon, config=config_nhits, loss=MQLoss(), num_samples=num_samples)]

    nf = NeuralForecast(models=models, freq="H")
    """
    nf.fit(df=df)
    Y_hat_df = nf.predict()
    Y_hat_df.head()
    
    """
    Y_hat_df = nf.cross_validation(df=df, n_windows=n_windows)

    # 儲存模型
    # [0] represents the first index of the modes_list, it's AutoLSTM
    nf.save(
        path=f"{checkpointsPath}/{model}_day{horizon}",
        model_index=[0],
        overwrite=True,
        save_dataset=True,
    )
    # [0] represents the best LSTM parameters by previous training
    best_config = nf.models[0].results.get_best_result().config
    print(f"best_config=\n{best_config}")
    with open(
        f"{checkpointsPath}/{model}_day{horizon}/best_{model}_config.json", "w"
    ) as f:
        f.write(str(best_config))
    return Y_hat_df


def model_evaluation(trained_model, trained_model_time, horizon, resultPath):
    path = resultPath
    model_list = []
    MAPE_result = []
    RMSE_result = []
    Training_time = []
    for model in trained_model:
        print(f"Evaluating {model} model")
        Y_hat_df = trained_model.get(model)
        Train_time = trained_model_time.get(model)
        # 顯示結果
        Y_hat_df.columns = Y_hat_df.columns.str.replace("-median", "")
        print(Y_hat_df)
        # 儲存預測結果
        Y_hat_df.to_csv(f"{path}/{model}_result_day{horizon}.csv", index=False)
        # 輸出Evaluation結果
        model_list.append(model)
        tag = f"Auto{model}"
        MAPE = round(mape(Y_hat_df["y"], Y_hat_df[tag]), 4)
        MAPE_result.append(MAPE)
        RMSE = round(rmse(Y_hat_df["y"], Y_hat_df[tag]), 4)
        RMSE_result.append(RMSE)
        Training_time.append(round(Train_time, 2))
    evaluation_result = {
        "Model": model_list,
        "MAPE": MAPE_result,
        "RMSE": RMSE_result,
        "Training_Time": Training_time,
    }
    result_df = pd.DataFrame(evaluation_result)
    print(result_df)
    result_df.to_csv(f"{path}/evaluation_day{horizon}.csv", index=False)


if __name__ == "__main__":
    filePath_List = [
        # "./dataset/2023_01_2024_06訂單彙整分店商品週期小資料_光明店.csv",
        # "./dataset/2023_01_2024_06訂單彙整分店商品週期小資料_大埔店.csv",
        # "./dataset/2023_01_2024_06訂單彙整分店商品週期小資料_大墩店.csv",
        # "./dataset/2023_01_2024_06訂單彙整分店商品週期小資料_朝富店.csv",
        # "./dataset/2023_01_2024_06訂單彙整分店商品週期小資料_東山店.csv",
        # "./dataset/2023_01_2024_06訂單彙整分店商品週期小資料_東興店.csv",
        # "./dataset/2023_01_2024_06訂單彙整分店商品週期小資料_永春店.csv",
        # "./dataset/2023_01_2024_06訂單彙整分店商品週期小資料_河南店.csv",
        #"./dataset/2023_01_2024_06訂單彙整分店商品週期小資料_經國店.csv",
        #"./dataset/2023_01_2024_06訂單彙整分店商品週期小資料_黎明店.csv"
        "./dataset/store_test.csv"
    ]
    # filePath_List=["./dataset/NTU/test/test.csv","./dataset/NTU/test/test_all.txt"]
    # filePath='./dataset/NTU/181697_Top70.csv'
    # filePath='./dataset/NTU/test/test_all.txt'

    # 預測模型
    # modelList = ["LSTM", "PatchTST", "NHITS"]
    modelList = ["Autoformer"]
    # 預測時間參數
    horizon_for_testing = 2

    # cross_validation 參數
    # n_windows = The number of windows to evaluate
    n_windows = 1
    # The number of samples, num_samples, is a crucial parameter! Larger values will usually produce better results as we explore more configurations in the search space, but it will increase training times. Larger search spaces will usually require more samples. As a general rule, we recommend setting num_samples higher than 20.
    num_samples = 25

    for filePath in filePath_List:
        for horizon in range(2, horizon_for_testing + 1):
            # ds使用datetime格式
            # df = pd.read_csv(filePath, header=None, names=["unique_id", "datetime", "y"])
            # df = df.drop('datetime', axis=1) # where 1 is the axis number (0 for rows and 1 for columns.)

            # ds使用datetime格式 -> 連續自然數格式
            # df = pd.read_csv(
            #     filePath, header=None, names=["unique_id", "datetime", "y"]
            # )
            # df = insert_ds(df)

            # ds使用連續自然數
            print(filePath)
            df = pd.read_csv(filePath,
                             header=None, skiprows=1, names=["unique_id", "ds", "y"])
            print(df.head())

            # 獲取檔案名稱
            # file name with extension
            file_name = os.path.basename(filePath)
            # file name without extension
            file_name = os.path.splitext(file_name)[0]
            print(f"Training: {file_name}")

            # 建立存放結果的資料夾
            resultPath = f"./result/{file_name}"
            if not os.path.isdir(resultPath):
                os.mkdir(resultPath)
            checkpointsPath = f"./checkpoints/{file_name}"
            if not os.path.isdir(checkpointsPath):
                os.mkdir(checkpointsPath)

            # 訓練模型
            trained_model = {}
            trained_model_time = {}
            for model in modelList:
                start_time = time.time()
                Y_hat_df = model_training(
                    df, horizon, model, n_windows, num_samples, checkpointsPath
                )
                end_time = time.time()
                execution_time = end_time - start_time
                trained_model[model] = Y_hat_df
                trained_model_time[model] = execution_time

            # 評估模型
            model_evaluation(trained_model, trained_model_time,
                             horizon, resultPath)
