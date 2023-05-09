# reference: https://github.com/Nixtla/neuralforecast/blob/main/nbs/examples/Getting_Started_complete.ipynb
import pandas as pd
from ray import tune
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS, AutoLSTM
from neuralforecast.losses.pytorch import MQLoss

# read parquest file
Y_df = pd.read_parquet('./dataset/m4-hourly.parquet')
# dump to csv file
# Y_df.to_csv("out.csv", encoding='utf-8', index=False)
# print(Y_df.head())

config_nhits = {
    "input_size": tune.choice([48, 48 * 2, 48 * 3, 48 * 5]),  # Length of input window
    "n_blocks": 5 * [1],  # Length of input window
    "mlp_units": 5 * [[64, 64]],  # Length of input window
    "n_pool_kernel_size": tune.choice([5 * [1], 5 * [2], 5 * [4],
                                       [8, 4, 2, 1, 1]]),  # MaxPooling Kernel size
    "n_freq_downsample": tune.choice([[8, 4, 2, 1, 1],
                                      [1, 1, 1, 1, 1]]),  # Interpolation expressivity ratios
    "learning_rate": tune.loguniform(1e-4, 1e-2),  # Initial Learning rate
    "scaler_type": tune.choice([None]),  # Scaler type
    "max_steps": tune.choice([1000]),  # Max number of training iterations
    "batch_size": tune.choice([32, 64, 128, 256]),  # Number of series in batch
    "windows_batch_size": tune.choice([128, 256, 512, 1024]),  # Number of windows in batch
    "random_seed": tune.randint(1, 20),  # Random seed
}
nf = NeuralForecast(
    models=[
        AutoNHITS(h=48, config=config_nhits, loss=MQLoss(), num_samples=5),
        AutoLSTM(h=48, loss=MQLoss(), num_samples=2),
    ],
    freq='H'
)
nf.fit(df=Y_df)
fcst_df = nf.predict()
fcst_df.columns = fcst_df.columns.str.replace('-median', '')
print(fcst_df.head())
# save result to csv
fcst_df.to_csv("results.csv", encoding='utf-8')
# save trained model
nf.save(path='./checkpoints/', overwrite=True)

