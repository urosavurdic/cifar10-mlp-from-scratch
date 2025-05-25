from flask import Flask, request, jsonify
import pickle
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from neural_network import NeuralNetwork3Layer

with open('C:/Users/Uros/Desktop/image_classifier/models/model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Extract the image array from the JSON — key 'image'
    img = np.array(data['image'], dtype=np.float32)  # shape (32, 32, 3)

    # Normalize pixel values to [0, 1]
    img /= 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)  # shape (1, 32, 32, 3)

    # Make prediction
    prediction = model.predict(img)

    # Return prediction as list
    return jsonify({'prediction': prediction.tolist()})