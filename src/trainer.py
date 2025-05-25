import numpy as np
from neural_network import NeuralNetwork3Layer
import pickle
from evaluator import evaluate_accuracy
import os

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, wait=5):
    """
    Train a model with given training and testing data over given number of epochs
    Function has early stopping criterion and saves best model
    """
    best_loss = float('inf')
    wait_counter = 0
    save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')

    for epoch in range(epochs):
        # forward pass
        y_pred_train = model.forward(X_train)
        #loss_train = model.loss(y_train, y_pred_train)

        # backward pass
        model.backward(X_train, y_train, y_pred_train)

        # evaluate on test set
        probs_test = model.forward(X_test)
        loss_test = model.loss(y_test, probs_test)
        prediction = np.argmax(probs_test, axis=1)

        accuracy = evaluate_accuracy(y_test, prediction)


        print(f"Epoch {epoch + 1}/{epochs}  - Loss: {float(loss_test):.4f} - Test Accuracy: {accuracy:.4f}")

        if loss_test < best_loss:
            best_loss = loss_test
            wait_counter = 0

            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model improved — saved to {save_path}")

        else:
            wait_counter += 1

        if wait_counter >= wait:
            print(f"Early stopping at epoch {epoch + 1}. Best Test Loss: {best_loss:.4f}")
            break

        print(f"Training complete. Best model saved to {save_path}")