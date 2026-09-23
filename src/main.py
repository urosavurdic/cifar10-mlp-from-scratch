"""
Trains the network end to end and reports test accuracy once.

Run from the repository root:

    python -m src.main
"""
from .data_loader import load_cifar10_data
from .evaluator import (
    evaluate_accuracy,
    plot_confusion_matrix,
    print_classification_report,
)
from .neural_network import NeuralNetwork3Layer
from .preprocessor import preprocess_data
from .trainer import train_model, train_val_split


def main():
    x_train_raw, y_train, x_test_raw, y_test = load_cifar10_data()

    x_train, x_test = preprocess_data(x_train_raw, x_test_raw)

    # 32 * 32 * 3 = 3072 inputs, one per colour channel per pixel
    model = NeuralNetwork3Layer(
        input_size=3072,
        output_size=10,
        hidden_size1=256,
        hidden_size2=128,
        learning_rate=0.01,
        seed=0,
    )

    x_tr, y_tr, x_val, y_val = train_val_split(x_train, y_train, val_fraction=0.1)
    train_model(model, x_tr, y_tr, x_val, y_val, epochs=100, patience=5)

    # The test set is touched once, here, after model selection is finished.
    predictions = model.predict(x_test)
    print(f"\nTest accuracy: {evaluate_accuracy(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(print_classification_report(y_test, predictions))
    plot_confusion_matrix(y_test, predictions)


if __name__ == "__main__":
    main()
