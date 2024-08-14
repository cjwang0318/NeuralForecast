from flask import Flask, jsonify, request
from threading import Thread
from forecasting import do_forecasting, convert_json2df
import time
import pandas as pd

app = Flask(__name__)

# Status dictionary to store the status of each request
status = {"state": "idle", "result": None}


def generate_results(sessionID, horizon, data, message):
    global status
    status["state"] = "processing"
    # Simulate image generation process
    # time.sleep(10)  # Replace this with the actual image generation logic
    result = do_forecasting(sessionID, horizon, data, message)
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
    # print(json_request)
    sessionID, horizon, data, message = convert_json2df(json_request)
    thread = Thread(target=generate_results, args=(
        sessionID, horizon, data, message))
    thread.start()

    return jsonify({"message": "Forecasting started"}), 202


@app.route('/status', methods=['GET'])
def status_route():
    global status
    return jsonify(status), 200


if __name__ == '__main__':
    app.run(host='192.168.50.26', port=8000, debug=True)
