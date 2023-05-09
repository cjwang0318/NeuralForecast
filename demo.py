import sys
import numpy as np
import pandas as pd
# from IPython.display import display, Markdown

import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.utils import AirPassengersDF

# Split data and declare panel dataset
Y_df = AirPassengersDF
"""
     unique_id         ds      y
0          1.0 1949-01-31  112.0
1          1.0 1949-02-28  118.0
"""
Y_train_df = Y_df[Y_df.ds <= '1959-12-31']  # 132 train
Y_test_df = Y_df[Y_df.ds > '1959-12-31']  # 12 test

# Fit and predict with N-BEATS and N-HiTS models
horizon = len(Y_test_df) # set prediction length
"""
h: int, forecast horizon.
input_size: int, considered autorregresive inputs (lags), y=[1,2,3,4] input_size=2 -> lags=[1,2].
"""
models = [NBEATS(input_size=2 * horizon, h=horizon, max_epochs=50),
          NHITS(input_size=2 * horizon, h=horizon, max_epochs=50)]
nforecast = NeuralForecast(models=models, freq='M')
nforecast.fit(df=Y_train_df)
Y_hat_df = nforecast.predict().reset_index()
"""
    unique_id         ds      NBEATS       NHITS
0         1.0 1960-01-31  396.054321  421.363098
1         1.0 1960-02-29  421.144104  450.626495
"""
# save result to csv
#Y_hat_df.to_csv("out.csv", encoding='utf-8', index=False)

# save trained model
#nforecast.save(path='./checkpoints/test_run/')

# load trained model
#nforecast = NeuralForecast.load(path='./checkpoints/test_run/')

# Plot predictions
fig, ax = plt.subplots(1, 1, figsize=(20, 7))
Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])
"""
   unique_id         ds      y      NBEATS       NHITS
0         1.0 1960-01-31  417.0  396.054321  421.363098
1         1.0 1960-02-29  391.0  421.144104  450.626495
"""
Y_hat_df.to_csv("out.csv", encoding='utf-8', index=False)

plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')
"""
            unique_id      y      NBEATS       NHITS
ds                                                  
1949-01-31        1.0  112.0         NaN         NaN
1949-02-28        1.0  118.0         NaN         NaN
1949-03-31        1.0  132.0         NaN         NaN
1949-04-30        1.0  129.0         NaN         NaN
1949-05-31        1.0  121.0         NaN         NaN
...               ...    ...         ...         ...
1960-08-31        1.0  606.0  605.867798  626.619568
1960-09-30        1.0  508.0  529.818298  550.767883
1960-10-31        1.0  461.0  452.984619  467.683472
1960-11-30        1.0  390.0  416.831482  437.846283
1960-12-31        1.0  432.0  446.840424  439.903412
"""
#evaluation
from datasetsforecast.evaluation import accuracy
from datasetsforecast.losses import rmse, mape
evaluation_df = accuracy(Y_hat_df, [rmse, mape], agg_by=['unique_id'])
print(evaluation_df)

plot_df[['y', 'NBEATS', 'NHITS']].plot(ax=ax, linewidth=2)
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
plt.show()
