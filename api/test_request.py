import requests
import numpy as np

# Generate a dummy CIFAR-10 image (32x32 RGB, values 0-255)
dummy_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8).tolist()

# Prepare the payload as JSON
payload = {
    "image": dummy_image
}

# Send POST request to the Flask app
response = requests.post('http://127.0.0.1:5000/predict', json=payload)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
