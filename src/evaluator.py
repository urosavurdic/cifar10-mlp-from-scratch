import os

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

DEFAULT_PLOT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "confusion_matrix.png"
)


def evaluate_accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return accuracy_score(y_true, y_pred)


def print_classification_report(y_true, y_pred):
    """Per-class precision, recall and F1."""
    return classification_report(y_true, y_pred, target_names=CLASS_NAMES)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Writes a labelled confusion matrix to disk and returns its path.

    matplotlib and seaborn are imported here rather than at module level so
    that training and serving do not depend on a plotting stack. savefig
    must come before close(), and the Agg backend means there is no
    interactive window to show.
    """
    import matplotlib
    matplotlib.use("Agg")
    import seaborn as sns
    from matplotlib import pyplot as plt

    save_path = save_path or DEFAULT_PLOT_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to {save_path}")
    return save_path
