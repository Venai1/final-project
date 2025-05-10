
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(cm, class_names):
    """
    Visualizes the confusion matrix using matplotlib.

    Args:
        cm (np.ndarray): Confusion matrix (n_classes x n_classes)
        class_names (list): List of class labels (e.g., [0, 1, ..., 9])
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.title("Confusion Matrix")
    plt.show()

def compute_per_class_accuracy(cm):
    """
    Computes per-class accuracy from the confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix (n_classes x n_classes)
        
    Returns:
        per_class_accuracy (dict): key = class label, value = accuracy %
    """
    n_classes = cm.shape[0]
    per_class_accuracy = {}
    
    for i in range(n_classes):
        correct = cm[i, i]
        total = cm[i, :].sum()
        acc = correct / total if total > 0 else 0
        per_class_accuracy[i] = round(acc * 100, 2)  # as %
    
    return per_class_accuracy
