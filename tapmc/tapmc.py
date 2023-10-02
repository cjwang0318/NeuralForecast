import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os
from neuralforecast.core import NeuralForecast
from neuralforecast.models import PatchTST, LSTM
from neuralforecast.auto import AutoLSTM, AutoPatchTST, AutoNHITS
from neuralforecast.losses.pytorch import MQLoss, MAPE, HuberMQLoss, MAE
from neuralforecast.losses.numpy import mape, rmse
from ray import tune
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from ray.tune.search.hyperopt import HyperOptSearch
import json
import config as cf

def insert_ds(df):
    uids = df['unique_id'].unique()  # Select ids
    df_return = pd.DataFrame()
    for id in uids:
        df_temp = df.query('unique_id in @id').reset_index(drop=True)
        start = 1
        df_temp.insert(2, 'ds', range(start, start + df_temp.shape[0]))
        #print(Y_df.shape)
        #print(Y_df.head(20))
        df_return = pd.concat([df_return, df_temp], axis=0, ignore_index=True)
    #print(df_return.tail())
    return df_return

if __name__ == '__main__':
    # ds使用連續自然數
    #df = pd.read_csv('./dataset/NTU/test.csv', header=None, names=["unique_id", "datetime", "y"])
    #df = insert_ds(df)
    #df = df.drop('datetime', axis=1) # where 1 is the axis number (0 for rows and 1 for columns.)
    
    # 使用資料原本格式
    df = pd.read_csv('./dataset/NTU/test_all.txt', header=None, names=["unique_id", "ds", "y"])
    # 修改成 pandas的datetime格式
    #df['ds'] = pd.to_datetime(df['ds'])  # transform to datetime formate
    
    #print(df.head())

    #預測時間參數
    horizon = 3
    # cross_validation 參數
    n_windows = 1


    

    # Utilizing Auto-models for parameters tuning by random grid search, and choose the best parameters by cross validation

    models = [
        AutoLSTM(h=horizon, config=cf.config_lstm, loss=MQLoss(), num_samples=20),
        #AutoPatchTST(h=horizon, config=config_patchtst, loss=MQLoss(), num_samples=2),
        #AutoNHITS(h=horizon, config=config_nhits, loss=MQLoss(), num_samples=2)
    ]

    nf = NeuralForecast(models=models, freq='H')
    """
    nf.fit(df=df)
    Y_hat_df = nf.predict()
    Y_hat_df.head()
    
    """
    Y_hat_df = nf.cross_validation(df=df, n_windows=n_windows)
    
    # 儲存模型
    # [0] represents the first index of the modes_list, it's AutoLSTM
    nf.save(path='./checkpoints/AutoLSTM_day3', model_index=[0], overwrite=True, save_dataset=True)
    # [0] represents the best LSTM parameters by previous training 
    LSTM_best_config=nf.models[0].results.get_best_result().config
    print(f"LSTM_best_config=\n{LSTM_best_config}")
    with open('./checkpoints/AutoLSTM_day3/best_LSTM_config.json', 'w') as f:
        f.write(str(LSTM_best_config))
    
    #顯示結果
    Y_hat_df.columns = Y_hat_df.columns.str.replace('-median', '')
    print(Y_hat_df)
    
    # 儲存預測結果
    Y_hat_df.to_csv("result.csv",index=False)
    
    # 輸出Evaluation結果
    modle_list=["LSTM"]
    MAPE_result=[]
    RMSE_result=[]
    MAPE=round(mape(Y_hat_df['y'], Y_hat_df['AutoLSTM']),4)
    MAPE_result.append(MAPE)
    RMSE=round(rmse(Y_hat_df['y'], Y_hat_df['AutoLSTM']),4)
    RMSE_result.append(RMSE)
    evaluation_result={
        "Model":modle_list,
        "MAPE":MAPE_result,
        "RMSE":RMSE_result
         }
    result_df=pd.DataFrame(evaluation_result)
    print(result_df)
    result_df.to_csv("evaluation.csv",index=False)


