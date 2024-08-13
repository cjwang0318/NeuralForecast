from flask import Flask, jsonify, request
from threading import Thread
from forecasting import do_forecasting
import time
import pandas as pd

app = Flask(__name__)

# Status dictionary to store the status of each request
status = {"state": "idle", "result": None}


def convert_json2df(json_str):

    # Extract metadata if needed
    session_id = json_str.get('sessionID')
    horizon = json_str.get('horizon')

    # Extract the main data
    data_records = json_str['data']

    # Convert the data to a DataFrame
    df = pd.DataFrame(data_records)

    # print(f"Session ID: {session_id}")
    # print("DataFrame:")
    # print(df)
    return session_id, horizon, df


def generate_results(sessionID, horizon, data):
    global status
    status["state"] = "processing"
    # Simulate image generation process
    # time.sleep(10)  # Replace this with the actual image generation logic
    result = do_forecasting(sessionID, horizon, data)
    status["state"] = "complete"
    status["result"] = result


@app.route('/forecasting', methods=['POST'])
def forecasting_route():
    global status
    if status["state"] == "processing":
        return jsonify({"message": "Forecasting is already in progress"}), 409

    # Reset result to None before starting new generation
    status["result"] = None

    # Parse JSON data from the request
    json_request = request.get_json()
    #print(json_request)
    sessionID, horizon, data = convert_json2df(json_request)
    thread = Thread(target=generate_results, args=(sessionID, horizon, data))
    thread.start()

    return jsonify({"message": "Forecasting started"}), 202


@app.route('/status', methods=['GET'])
def status_route():
    global status
    return jsonify(status), 200


if __name__ == '__main__':
    app.run(host='192.168.50.26', port=8000, debug=True)
