from flask import Flask, jsonify, request
from threading import Thread
from forecasting import forecasting
import time

app = Flask(__name__)

# Status dictionary to store the status of each request
status = {"state": "idle", "result": None}


def generate_results():
    global status
    status["state"] = "processing"
    # Simulate image generation process
    time.sleep(10)  # Replace this with the actual image generation logic
    result={"1":1.0, "2":2.0, "3":3.0}
    status["state"] = "complete"
    status["result"] = result


@app.route('/forecasting', methods=['POST'])
def forecasting_route():
    global status
    if status["state"] == "processing":
        return jsonify({"message": "Forecasting is already in progress"}), 409

    # Reset result to None before starting new generation
    status["result"] = None

    thread = Thread(target=generate_results)
    thread.start()

    return jsonify({"message": "Forecasting started"}), 202


@app.route('/status', methods=['GET'])
def status_route():
    global status
    return jsonify(status), 200


if __name__ == '__main__':
    app.run(host='192.168.50.26', port=5000, debug=True)
