import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import (
    mae, mse, rmse,              # unscaled errors
    mape, smape,                 # percentage errors
    mase,                        # scaled error
    quantile_loss, mqloss        # probabilistic errors
)
df = pd.read_csv('out.csv')
predict_value=df.iloc[:,2].values
ground_truth=df.iloc[:,3].values
#y_true = Y_test_df.y.values
#y_hat = Y_hat_df['NHITS'].values

result  = rmse(predict_value, ground_truth)
print("rmse="+str(result))

result  = mape(predict_value, ground_truth)
print("mape="+str(result))