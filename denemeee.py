import requests

url = "http://127.0.0.1:8080/predict"
data = {
    "fever": "Yes",
    "cough": "No",
    "fatigue": "Yes",
    "breathing": "No",
    "age": 30,
    "gender": "Male",
    "bloodPressure": "Normal",
    "cholesterol": "High"
}

response = requests.post(url, json=data)
print("Response:", response.json())
