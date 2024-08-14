import requests
import json


def client_send(url, json_str, token):

    # Example headers (you might need to customize these based on your API's requirements)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        # Replace with your actual token if needed
        "Authorization": f"Bearer {token}"
    }
    # print(headers)
    # Example payload (data to be sent in the request)
    payload = json_str

    # Making the API request
    response = requests.post(url, json=payload, headers=headers)

    # Checking if the request was successful
    if response.status_code == 200:
        # Parsing the JSON response
        data = response.json()
        print("Response Data:", data)
        return data
    else:
        print(f"Request failed with status code {response.status_code}")
        print("Response:", response.text)


def get_Authorization():
    # Base URL of the API
    base_url = "https://sales-forecasting-stg.dasgo.com.tw/api"
    # Example endpoint path (you should replace this with the actual path from your Swagger documentation)
    endpoint = "/ItriSys/ApiLogin"
    # Full URL for the API request
    url = f"{base_url}{endpoint}"
    str = {
        "userId": "itri",
        "password": "02750963"
    }
    output = client_send(url, str, "")
    # print(output)
    token = output.get("token")
    # print(token)
    if token:
        print("Token received:", token)
        return token
    else:
        print("Failed to retrieve token")
        return None


def update_results(token, json_result):
    # Base URL of the API
    base_url = "https://sales-forecasting-stg.dasgo.com.tw/api"

    # Example endpoint path (you should replace this with the actual path from your Swagger documentation)
    endpoint = "/Order/UploadForecastSalesResult"

    # Full URL for the API request
    url = f"{base_url}{endpoint}"

    output = client_send(url, json_result, token)


if __name__ == "__main__":
    token = get_Authorization()
    print(token)
    json_result = {"message": "abced", "sessionID": "abced",
                   "data": "[{\"unique_id\":\"11\",\"ds\":1,\"val\":-2},{\"unique_id\":\"11\",\"ds\":2,\"val\":0},{\"unique_id\":\"11\",\"ds\":3,\"val\":-3},{\"unique_id\":\"31\",\"ds\":1,\"val\":70},{\"unique_id\":\"31\",\"ds\":2,\"val\":70},{\"unique_id\":\"31\",\"ds\":3,\"val\":70}]"}
    update_results(token, json_result)
