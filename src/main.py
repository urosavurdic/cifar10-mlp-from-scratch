import numpy as np
from data_loader import load_cifar10_data
from neural_network import NeuralNetwork3Layer
from trainer import train_model
from evaluator import print_classification_report, plot_confusion_matrix
from preprocessor import preprocess_data

# load data
X_train, y_train, X_test, y_test = load_cifar10_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

# preprocess data (reshape + normalize)
x_train, x_test = preprocess_data(X_train, X_test)

# construct model (input size = 3072)
model = NeuralNetwork3Layer(
    input_size=3072,
    output_size=10,
    hidden_size1=256,
    hidden_size2=128,
    learning_rate=0.01
)

# train model
train_model(model, x_train, y_train, x_test, y_test, epochs=100)

# evaluation
predictions = model.predict(x_test)

print("\nFinal Classification Report:")
print(print_classification_report(y_test, predictions))
plot_confusion_matrix(y_test, predictions)
