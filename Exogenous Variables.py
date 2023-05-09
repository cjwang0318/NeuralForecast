# https://nixtla.github.io/neuralforecast/examples/exogenous_variables.html
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast.auto import NHITS
from neuralforecast.core import NeuralForecast

df = pd.read_csv('./dataset/EPF_FR_BE.csv')
df['ds'] = pd.to_datetime(df['ds'])
# print(df.head())
static_df = pd.read_csv('./dataset/EPF_FR_BE_static.csv')

futr_df = pd.read_csv('./dataset/EPF_FR_BE_futr.csv')
futr_df['ds'] = pd.to_datetime(futr_df['ds'])

horizon = 24  # day-ahead daily forecast
models = [NHITS(h=horizon,
                input_size=5 * horizon,
                futr_exog_list=['gen_forecast', 'week_day'],  # <- Future exogenous variables
                hist_exog_list=['system_load'],  # <- Historical exogenous variables
                stat_exog_list=['market_0', 'market_1'],  # <- Static exogenous variables
                scaler_type='robust')]
nf = NeuralForecast(models=models, freq='H')
nf.fit(df=df,
       static_df=static_df)
Y_hat_df = nf.predict(futr_df=futr_df)
print(Y_hat_df.head())
