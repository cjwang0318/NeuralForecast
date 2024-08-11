import json
from ray import tune
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS, AutoLSTM
from neuralforecast.losses.pytorch import MQLoss
from datasetsforecast.losses import mse, mae, rmse, mape
from datasetsforecast.evaluation import accuracy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_demo_data(path):
    Y_df = pd.read_parquet(path)
    # print(Y_df.head())
    # Select 10 ids to make the example faster
    uids = Y_df['unique_id'].unique()[:1]
    Y_df = Y_df.query('unique_id in @uids').reset_index(drop=True)
    # print(Y_df.head())
    return Y_df


def load_csv_data(filePath):
    Y_df = pd.read_csv(filePath,
                       header=None, skiprows=1, names=["unique_id", "ds", "y"])
    # Y_df['ds'] = pd.to_datetime(Y_df['ds'])
    # Select 1 ids to make the example faster
    # uids = Y_df['unique_id'].unique()[:1]
    # Y_df = Y_df.query('unique_id in @uids').reset_index(drop=True)
    # print(Y_df)
    return Y_df


def save_and_load_df(cv_df):
    # Save the DataFrame
    cv_df.to_csv('cv_df.csv', index=False)
    # Reload the DataFrame
    cv_df = pd.read_csv('cv_df.csv')
    # Print the reloaded DataFrame
    # print(cv_df)
    return cv_df


config_nhits = {
    # Length of input window
    # "input_size": tune.choice([48, 48*2, 48*3]),
    "start_padding_enabled": True,
    # Length of input window
    "n_blocks": 5*[1],
    # Length of input window
    "mlp_units": 5 * [[64, 64]],
    "n_pool_kernel_size": tune.choice([5*[1], 5*[2], 5*[4],
                                      [8, 4, 2, 1, 1]]),            # MaxPooling Kernel size
    "n_freq_downsample": tune.choice([[8, 4, 2, 1, 1],
                                      [1, 1, 1, 1, 1]]),            # Interpolation expressivity ratios
    # Initial Learning rate
    "learning_rate": tune.loguniform(1e-4, 1e-2),
    # Scaler type
    "scaler_type": tune.choice([None]),
    # Max number of training iterations
    "max_steps": tune.choice([1000]),
    # Number of series in batch
    "batch_size": tune.choice([1, 4, 10]),
    # Number of windows in batch
    "windows_batch_size": tune.choice([128, 256, 512]),
    # Random seed
    "random_seed": tune.randint(1, 20),
}

config_lstm = {
    # Length of input window
    # "input_size": tune.choice([48, 48*2, 48*3]),
    # Hidden size of LSTM cells
    "encoder_hidden_size": tune.choice([64, 128]),
    # Number of layers in LSTM
    "encoder_n_layers": tune.choice([2, 4]),
    # Initial Learning rate
    "learning_rate": tune.loguniform(1e-4, 1e-2),
    "scaler_type": tune.choice(['robust']),                   # Scaler type
    # Max number of training iterations
    "max_steps": tune.choice([500, 1000]),
    # Number of series in batch
    "batch_size": tune.choice([1, 4]),
    "random_seed": tune.randint(1, 20),                       # Random seed
}


def evaluation_average(df):
    # 根據不同prediction algorithms去平均不同的evaluation metrics values
    # Group by the 'metric' column and calculate the mean for 'AutoNHITS' and 'AutoLSTM'
    average_metrics = df.groupby('metric')[['AutoNHITS', 'AutoLSTM']].mean()
    # Print the resulting DataFrame
    # print(average_metrics)
    average_metrics.to_csv('./result/average_metrics_df.csv')

    # 根據best_model去去平均不同的evaluation metrics values
    # Select the scores based on the best_model column
    df['best_score'] = df.apply(lambda row: row[row['best_model']], axis=1)
    # Group by the 'metric' column and calculate the average of the selected scores
    best_model_average_scores = df.groupby('metric')['best_score'].mean()
    # print(best_model_average_scores)
    best_model_average_scores.to_csv(
        './result/best_model_average_scores.csv')


def evaluation_summary(evaluation_df):
    summary_df = evaluation_df.groupby(
        ['metric', 'best_model']).size().sort_values().to_frame()
    summary_df = summary_df.reset_index()
    summary_df.columns = ['metric', 'model', 'nr. of unique_ids']
    print(summary_df)


def get_best_model_forecast(forecasts_df, evaluation_df, metric):
    df = forecasts_df.set_index('ds', append=True).stack(
    ).to_frame().reset_index(level=2)  # Wide to long
    df.columns = ['model', 'best_model_forecast']
    df = df.join(evaluation_df.query(
        'metric == @metric').set_index('unique_id')[['best_model']])
    df = df.query(
        'model.str.replace("-lo-90|-hi-90", "", regex=True) == best_model').copy()
    df.loc[:, 'model'] = [model.replace(
        bm, 'best_model') for model, bm in zip(df['model'], df['best_model'])]
    df = df.drop(columns='best_model').set_index(
        'model', append=True).unstack()
    df.columns = df.columns.droplevel()
    df = df.reset_index(level=1)
    return df


def return_json_result(df):
    # Reset the `ds` values starting from 1
    df['ds'] = range(1, len(df) + 1)

    # Round the `best_model` values to 0 decimal places
    df['best_model'] = df['best_model'].round().astype(int)

    # Extract the `best_model` values as a dictionary with `ds` as the key
    best_model_dict = df.set_index('ds')['best_model'].to_dict()

    # Convert the dictionary to a JSON string
    best_model_json = json.dumps(best_model_dict, indent=4)

    # Output the JSON
    # print(best_model_json)

    return best_model_json


def forecasting(horizon, Y_df):
    config_nhits["input_size"] = tune.choice([3, 6, 8, 12])
    config_lstm["input_size"] = tune.choice([3, 6, 8, 12])
    # As a general rule, we recommend setting num_samples higher than 20.
    num_samples = 20

    # print(config_lstm)
    nf = NeuralForecast(
        models=[
            AutoNHITS(h=horizon, config=config_nhits,
                      loss=MQLoss(), num_samples=num_samples),
            AutoLSTM(h=horizon, config=config_lstm,
                     loss=MQLoss(), num_samples=num_samples),
        ],
        freq='H'
    )

    # Do cross_validation to find best parameters
    # cutoff: the last datestamp or temporal index for the n_windows. If n_windows = 1, then one unique cuttoff value, if n_windows = 2 then two unique cutoff values.
    cv_df = nf.cross_validation(Y_df, n_windows=1)
    cv_df.columns = cv_df.columns.str.replace('-median', '')
    # Save the cross_validation results
    # save_and_load_df(cv_df)

    # Evaluation forecasting performance
    evaluation_df = accuracy(
        cv_df, [mse, mae, rmse, mape], agg_by=['unique_id'])
    evaluation_df['best_model'] = evaluation_df.drop(
        columns=['metric', 'unique_id']).idxmin(axis=1)
    # print(evaluation_df.head())
    evaluation_df.to_csv('./result/evaluation_df.csv', index=False)

    # evaluation_summary(evaluation_df)
    evaluation_average(evaluation_df)

    # Create production-ready data frame with the best forecast for every unique_id
    # fcst_df = nf.predict()  # must run first: cv_df = nf.cross_validation(Y_df, n_windows=2)
    # fcst_df.columns = fcst_df.columns.str.replace('-median', '')
    # # print(fcst_df.head())
    # prod_forecasts_df = get_best_model_forecast(
    #     fcst_df, evaluation_df, metric='mse')
    # # print(prod_forecasts_df.head())
    # # save_and_load_df(prod_forecasts_df)

    # return_json_result(prod_forecasts_df)


if __name__ == "__main__":
    # train_data = load_csv_data('./dataset/test.csv')
    train_data = load_csv_data('./dataset/filtered_data_Seq.csv')
    forecasting(6, train_data)
    # cv_df = pd.read_csv('./result/evaluation_df.csv')
    # evaluation_average(cv_df)
