import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import seaborn as sns

def evaluate_accuracy(y_true, y_pred):
    """
    Simple accuracy evaluation function.
    """

    return accuracy_score(y_true, y_pred)

def print_classification_report(y_true, y_pred):
    """
    Prints classification report: precision, recall, F1 score
    """
    return classification_report(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix')
    plt.show()

    save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    plt.close()