import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import mape, rmse
from neuralforecast.losses.pytorch import MQLoss, MAPE, HuberMQLoss, MAE
from neuralforecast.auto import AutoLSTM, AutoPatchTST, AutoNHITS
from neuralforecast.models import PatchTST, LSTM
from neuralforecast.core import NeuralForecast
import json
import config as cf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from ray import tune
# from ray.tune.search.hyperopt import HyperOptSearch

# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def insert_ds(df):
    uids = df['unique_id'].unique()  # Select ids
    df_return = pd.DataFrame()
    for id in uids:
        df_temp = df.query('unique_id in @id').reset_index(drop=True)
        start = 1
        df_temp.insert(2, 'ds', range(start, start + df_temp.shape[0]))
        # print(Y_df.shape)
        # print(Y_df.head(20))
        df_return = pd.concat([df_return, df_temp], axis=0, ignore_index=True)
    # print(df_return.tail())
    return df_return


def model_training(horizon, model, n_windows, num_samples=2):
    # Utilizing Auto-models for parameters tuning by random grid search, and choose the best parameters by cross validation
    if model == "LSTM":
        models = [AutoLSTM(h=horizon, config=cf.config_lstm,
                           loss=MQLoss(), num_samples=num_samples)]
    elif model == "PatchTST":
        models = [AutoPatchTST(h=horizon, config=cf.config_patchtst,
                               loss=MQLoss(), num_samples=num_samples)]
    elif model == "NHITS":
        models = [AutoNHITS(h=horizon, config=cf.config_nhits,
                            loss=MQLoss(), num_samples=num_samples)]
    else:
        print("NO model is selected...")
        # models = [
        #     AutoLSTM(h=horizon, config=cf.config_lstm,
        #              loss=MQLoss(), num_samples=num_samples),
        #     AutoPatchTST(h=horizon, config=config_patchtst,
        #                  loss=MQLoss(), num_samples=num_samples),
        #     AutoNHITS(h=horizon, config=config_nhits, loss=MQLoss(), num_samples=num_samples)]

    nf = NeuralForecast(models=models, freq='H')
    """
    nf.fit(df=df)
    Y_hat_df = nf.predict()
    Y_hat_df.head()
    
    """
    Y_hat_df = nf.cross_validation(df=df, n_windows=n_windows)

    # 儲存模型
    # [0] represents the first index of the modes_list, it's AutoLSTM
    nf.save(path=f'./checkpoints/{model}_day{horizon}',
            model_index=[0], overwrite=True, save_dataset=True)
    # [0] represents the best LSTM parameters by previous training
    best_config = nf.models[0].results.get_best_result().config
    print(f"best_config=\n{best_config}")
    with open(f'./checkpoints/{model}_day{horizon}/best_{model}_config.json', 'w') as f:
        f.write(str(best_config))
    return Y_hat_df


def model_evaluation(trained_model):
    path = "./result"
    model_list = []
    MAPE_result = []
    RMSE_result = []
    if not os.path.isdir(path):
        os.mkdir(path)
    for model in trained_model:
        print(f"Evaluating {model} model")
        Y_hat_df = trained_model.get(model)
        # 顯示結果
        Y_hat_df.columns = Y_hat_df.columns.str.replace('-median', '')
        print(Y_hat_df)
        # 儲存預測結果
        Y_hat_df.to_csv(f"{path}/{model}_result.csv", index=False)
        # 輸出Evaluation結果
        model_list.append(model)
        tag = f"Auto{model}"
        MAPE = round(mape(Y_hat_df['y'], Y_hat_df[tag]), 4)
        MAPE_result.append(MAPE)
        RMSE = round(rmse(Y_hat_df['y'], Y_hat_df[tag]), 4)
        RMSE_result.append(RMSE)
    evaluation_result = {
        "Model": model_list,
        "MAPE": MAPE_result,
        "RMSE": RMSE_result
    }
    result_df = pd.DataFrame(evaluation_result)
    print(result_df)
    result_df.to_csv(f"{path}/evaluation.csv", index=False)


if __name__ == '__main__':
    # ds使用datetime格式
    # df = pd.read_csv('./dataset/NTU/test.csv', header=None, names=["unique_id", "datetime", "y"])
    # df = df.drop('datetime', axis=1) # where 1 is the axis number (0 for rows and 1 for columns.)

    # ds使用datetime格式 -> 連續自然數格式
    # df = pd.read_csv('./dataset/NTU/test.csv', header=None, names=["unique_id", "datetime", "y"])
    # df = insert_ds(df)

    # ds使用連續自然數
    df = pd.read_csv('./dataset/NTU/test/test_all.txt',
                     header=None, names=["unique_id", "ds", "y"])
    # print(df.head())

    # 預測模型
    modelList = ["LSTM"]

    # 預測時間參數
    horizon = 3
    # cross_validation 參數
    n_windows = 1

    # 訓練模型
    trained_model = {}
    for model in modelList:
        Y_hat_df = model_training(horizon, model, n_windows, 2)
        trained_model[model] = Y_hat_df

    # 評估模型
    model_evaluation(trained_model)
