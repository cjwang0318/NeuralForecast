import json
from ray import tune
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS, AutoLSTM
from neuralforecast.losses.pytorch import MQLoss
from datasetsforecast.losses import mse, mae, rmse
from datasetsforecast.evaluation import accuracy
import os
import ray
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
    # Select 2 ids to make the example faster
    uids = Y_df['unique_id'].unique()[:2]
    Y_df = Y_df.query('unique_id in @uids').reset_index(drop=True)
    # print(Y_df)
    return Y_df


def convert_df2json(json_path, json_output, df):
    # Convert DataFrame to JSON format
    json_output["data"] = json.loads(df.to_json(
        orient='records', indent=4))  # Data as JSON

    # Convert the Python dictionary to a pretty-printed JSON string
    pretty_json = json.dumps(json_output, indent=4)

    # Write the pretty-printed JSON to a file
    with open(json_path, 'w') as json_file:
        json_file.write(pretty_json)


def convert_json2df(json_str):

    # Extract metadata if needed
    session_id = json_str.get('sessionID')
    horizon = json_str.get('horizon')

    # Extract the main data
    data_records = json_str['data']

    # Convert the data to a DataFrame
    df = pd.DataFrame(data_records)

    # Group by 'unique_id' and count the rows for each 'unique_id'
    unique_id_counts = df.groupby('unique_id').size().reset_index(name='counts')
    summary_str = [
        f"unique_id {row['unique_id']} has {row['counts']} rows" for _, row in unique_id_counts.iterrows()]
    message = "; ".join(summary_str)
    # print(message)     
  
    return session_id, horizon, df, message


def save_and_load_df(cv_df):
    # Save the DataFrame
    cv_df.to_csv('cv_df.csv', index=True)
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

    # Reset the `ds` values starting from 1 within each unique_id group
    df['ds'] = df.groupby(df.index).cumcount() + 1
    # Round the `best_model` values to 0 decimal places
    df['best_model'] = df['best_model'].round().astype(int)

    # Initialize a list to hold the final output
    output_list = []

    # Create the desired structure
    for index, row in df.iterrows():
        entry = {
            "unique_id": str(index),
            "ds": int(row['ds']),
            "val": int(row['best_model'])
        }
        output_list.append(entry)

    # Convert the list to a JSON string
    output_json = json.dumps(output_list, separators=(',', ':'))

    # Output the JSON
    #print(output_json)

    return output_json


def forecasting(horizon, Y_df):
    config_nhits["input_size"] = tune.choice([2, 7, 14, 28, 56])
    config_lstm["input_size"] = tune.choice([2, 7, 14, 28, 56])
    # As a general rule, we recommend setting num_samples higher than 20.
    num_samples = 1

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
    evaluation_df = accuracy(cv_df, [mse, mae, rmse], agg_by=['unique_id'])
    evaluation_df['best_model'] = evaluation_df.drop(
        columns=['metric', 'unique_id']).idxmin(axis=1)
    # print(evaluation_df.head())
    # evaluation_summary(evaluation_df)

    # Create production-ready data frame with the best forecast for every unique_id
    fcst_df = nf.predict()  # must run first: cv_df = nf.cross_validation(Y_df, n_windows=2)
    fcst_df.columns = fcst_df.columns.str.replace('-median', '')
    # print(fcst_df.head())
    
    prod_forecasts_df = get_best_model_forecast(
        fcst_df, evaluation_df, metric='mse')
    #print(prod_forecasts_df.head())
    #save_and_load_df(prod_forecasts_df)

    best_model_json = return_json_result(prod_forecasts_df)
    return best_model_json


def do_forecasting(sessionID, horizon, data, message):

    #sessionID, horizon, data = convert_json2df(json_str)
    best_model_json = forecasting(horizon, data)
    json_str = {"message": message,
                "sessionID": sessionID, "data": best_model_json}
    print(f"sessionID={sessionID} is finished...")
    # Failed to connect to GCS within 60 seconds. GCS may have been killed 必需加上下面那行，把ray關閉
    ray.shutdown()
    return json.dumps(json_str)
    # Shutdown any existing Ray cluster
    
    

if __name__ == "__main__":
    # train_data = load_demo_data('../dataset/m4-hourly.parquet')
    # forecasting(3, train_data)

    # Generate testing data
    # train_data = load_csv_data('./dataset/store_test.csv')
    # sessionID = "abced"
    # json_dict = {"sessionID": sessionID, "horizon": 3}
    # convert_df2json("output.json", json_dict, train_data)

    # Testing forecasting function
    f = open('output.json')
    json_str = json.load(f)
    sessionID, horizon, data, message = convert_json2df(json_str)
    results = do_forecasting(sessionID, horizon, data, message)
    print(results)

    # cv_df = pd.read_csv('cv_df.csv')
    # best_model_json = return_json_result(cv_df)
    # print(best_model_json)
