# reference: https://github.com/Nixtla/neuralforecast/blob/main/nbs/examples/Getting_Started_complete.ipynb
import pandas as pd
from neuralforecast.auto import AutoNHITS, AutoLSTM
from ray import tune
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from datasetsforecast.losses import mse, mae, rmse
from datasetsforecast.evaluation import accuracy

# read parquest file
Y_df = pd.read_parquet('./dataset/m4-hourly.parquet')

config_nhits = {
    "input_size": tune.choice([48, 48*2, 48*3, 48*5]),              # Length of input window
    "n_blocks": 5*[1],                                              # Length of input window
    "mlp_units": 5 * [[64, 64]],                                  # Length of input window
    "n_pool_kernel_size": tune.choice([5*[1], 5*[2], 5*[4],
                                      [8, 4, 2, 1, 1]]),            # MaxPooling Kernel size
    "n_freq_downsample": tune.choice([[8, 4, 2, 1, 1],
                                      [1, 1, 1, 1, 1]]),            # Interpolation expressivity ratios
    "learning_rate": tune.loguniform(1e-4, 1e-2),                   # Initial Learning rate
    "scaler_type": tune.choice([None]),                             # Scaler type
    "max_steps": tune.choice([1000]),                               # Max number of training iterations
    "batch_size": tune.choice([32, 64, 128, 256]),                  # Number of series in batch
    "windows_batch_size": tune.choice([128, 256, 512, 1024]),       # Number of windows in batch
    "random_seed": tune.randint(1, 20),                             # Random seed
}
nf = NeuralForecast(
    models=[
        AutoNHITS(h=48, config=config_nhits, loss=MQLoss(), num_samples=5),
        AutoLSTM(h=48, loss=MQLoss(), num_samples=2),
    ],
    freq='H'
)
cv_df = nf.cross_validation(Y_df, n_windows=2)
cv_df.columns = cv_df.columns.str.replace('-median', '')

# save result to csv
cv_df.to_csv("cv_df.csv", encoding='utf-8')
# save trained model
nf.save(path='./checkpoints/', overwrite=True)

#evluation
evaluation_df = accuracy(cv_df, [mse, mae, rmse], agg_by=['unique_id'])
evaluation_df['best_model'] = evaluation_df.drop(columns=['metric', 'unique_id']).idxmin(axis=1)
print(evaluation_df.head())

# save result to csv
evaluation_df.to_csv("evaluation_df.csv", encoding='utf-8')

#Create a summary table with a model column and the number of series where that model performs best.
summary_df = evaluation_df.groupby(['metric', 'best_model']).size().sort_values().to_frame()

summary_df = summary_df.reset_index()
summary_df.columns = ['metric', 'model', 'nr. of unique_ids']
print(summary_df)

# save result to csv
cv_df.to_csv("summary_df.csv", encoding='utf-8')