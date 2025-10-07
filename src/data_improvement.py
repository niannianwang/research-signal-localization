import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def compute_removal_ratio(y_true, y_pred, target_accuracy):
    """
    Compute how much misclassified data needs to be removed to reach the desired accuracy.
    
    Parameters:
        y_true: Ground truth class labels (not one-hot)
        y_pred: Predicted class labels (not softmax)
        target_accuracy: Desired accuracy (e.g., 0.98)
    
    Returns:
        removal_ratio: Fraction of misclassified samples to remove
    """
    
    y_true = np.argmax(y_true, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total_correct = tn + tp
    total_incorrect = fp + fn

    # Total incorrect allowed to reach target accuracy
    target_total_incorrect = (total_correct / target_accuracy) - total_correct

    # How much to remove from current misclassified
    to_remove = total_incorrect - target_total_incorrect
    removal_ratio = to_remove / total_incorrect
    
    return max(0.0, min(removal_ratio, 1.0))

def remove_misclassified_data(X_test, y_test, y_pred, removal_ratio=0.5):
    """
    Remove a proportion of misclassified data while maintaining balance between false positives and negatives.
    
    Parameters:
        X_test: Test data
        y_test: True labels
        y_pred: Model predictions
        removal_ratio: Proportion of misclassified data to remove
        
    Returns:
        X_test_improved: Improved test data
        y_test_improved: Improved test labels
        removed_indices: Indices of removed samples
    """
    # Get confusion matrix
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    
    # Find misclassified indices
    misclassified_indices = np.where(y_true != y_pred)[0]
    fp_indices = misclassified_indices[y_true[misclassified_indices] == 0]
    fn_indices = misclassified_indices[y_true[misclassified_indices] == 1]
    
    # Calculate number of samples to remove from each category
    n_fp_remove = int(len(fp_indices) * removal_ratio)
    n_fn_remove = int(len(fn_indices) * removal_ratio)
    
    # Randomly select samples to remove
    np.random.seed(42)
    remove_FP_indices = np.random.choice(fp_indices, n_fp_remove, replace=False)
    remove_FN_indices = np.random.choice(fn_indices, n_fn_remove, replace=False)
    indices_to_remove = np.concatenate([remove_FP_indices, remove_FN_indices])
    
    # Create mask for data filtering indices to remove
    mask = np.ones(len(X_test), dtype=bool)
    mask[indices_to_remove] = False
    
    # Filter data
    X_test_improved = X_test[mask]
    y_test_improved = y_test[mask]
    
    return X_test_improved, y_test_improved, indices_to_remove

def compute_removal_ratio_doa(y_true, y_pred, target_accuracy):
    """
    Compute how much misclassified data needs to be removed to reach the desired accuracy.

    Parameters:
        y_true: Ground truth class indices (not one-hot)
        y_pred: Predicted class indices
        target_accuracy: Desired accuracy (e.g., 0.90)

    Returns:
        removal_ratio: Fraction of misclassified samples to remove
    """
    total = len(y_true)
    total_correct = np.sum(y_true == y_pred)
    total_incorrect = total - total_correct

    target_total_incorrect = (total_correct / target_accuracy) - total_correct

    # How much to remove from current misclassified
    to_remove = total_incorrect - target_total_incorrect
    removal_ratio = to_remove / total_incorrect

    return max(0.0, min(removal_ratio, 1.0))

def remove_misclassified_data_doa(X_test, y_test, y_pred, removal_ratio=0.5):
    """
    Remove a proportion of misclassified samples from the test set.

    Parameters:
        X_test: np.ndarray, test data
        y_test: np.ndarray, one-hot or class indices
        y_pred: np.ndarray, predicted class indices
        removal_ratio: float, proportion of misclassified samples to remove

    Returns:
        X_test_improved, y_test_improved, indices_removed
    """
    if y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    misclassified = np.where(y_true != y_pred)[0]
    n_remove = int(len(misclassified) * removal_ratio)

    np.random.seed(42)
    indices_to_remove = np.random.choice(misclassified, n_remove, replace=False)

    mask = np.ones(len(X_test), dtype=bool)
    mask[indices_to_remove] = False

    return X_test[mask], y_test[mask], indices_to_remove

def save_improved_dataset(X_train, X_test_improved, y_train, y_test_improved, 
                        signal2_start_times, removed_indices, filename='improved_signals.mat'):
    """
    Save the improved dataset to a .mat file.
    
    Parameters:
        X_train: Training data
        X_test_improved: Improved test data
        y_train: Training labels
        y_test_improved: Improved test labels
        signal2_start_times: Original signal2 start times
        filename: Output filename
    """
    mask = np.ones(len(signal2_start_times), dtype=bool)
    mask[removed_indices] = False
    signal2_start_times_improved = signal2_start_times[mask]

    save_dict = {
        'X_train': X_train,
        'X_test': X_test_improved,
        'y_train': y_train,
        'y_test': y_test_improved,
        'signal2_start_times': signal2_start_times_improved
    }
    
    scipy.io.savemat(filename, save_dict)
    print(f"Improved dataset saved to {filename}") 

def save_improved_doa_dataset(X_train, X_test_improved, y_train, y_test_improved, filename='improved_doa_signals.mat'):
    """
    Save the improved DoA dataset to a .mat file.

    Parameters:
        X_train: np.ndarray — Training data
        X_test_improved: np.ndarray — Filtered test data
        y_train: np.ndarray — Training labels
        y_test_improved: np.ndarray — Filtered test labels
        filename: str — Output filename
    """
    save_dict = {
        'X_train': X_train,
        'X_test': X_test_improved,
        'y_train': y_train,
        'y_test': y_test_improved,
    }

    scipy.io.savemat(filename, save_dict)
    print(f"Improved DoA dataset saved to {filename}")

def plot_confusion_matrix(y_true, y_pred, title="GLRT Confusion Matrix"):
    """
    Plot the confusion matrix for the given true and predicted labels.
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        title: Title for the plot
    """
    
    cm = confusion_matrix(np.argmax(y_true, axis=1),y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()
