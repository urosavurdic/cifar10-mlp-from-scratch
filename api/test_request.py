"""Sends one random image to a running server. Start `python -m api.app` first."""
import numpy as np
import requests

dummy_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8).tolist()

response = requests.post("http://127.0.0.1:5000/predict", json={"image": dummy_image})

print("Status code:", response.status_code)
print("Response JSON:", response.json())
