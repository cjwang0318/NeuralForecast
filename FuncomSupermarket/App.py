from flask import Flask, jsonify, request
from threading import Thread
import time

app = Flask(__name__)

# Status dictionary to store the status of each request
status = {"state": "idle"}


def generate_image():
    global status
    status["state"] = "processing"
    # Simulate image generation process
    time.sleep(10)  # Replace this with the actual image generation logic
    status["state"] = "complete"


@app.route('/generate_image', methods=['POST'])
def generate_image_route():
    global status
    if status["state"] == "processing":
        return jsonify({"message": "Image generation is already in progress"}), 409

    thread = Thread(target=generate_image)
    thread.start()

    return jsonify({"message": "Image generation started"}), 202


@app.route('/status', methods=['GET'])
def status_route():
    global status
    return jsonify(status), 200


if __name__ == '__main__':
    app.run(host='192.168.50.26', port=5000, debug=True)
