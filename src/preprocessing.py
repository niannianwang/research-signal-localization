import numpy as np
import scipy.io
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath='processed_signals.mat', test_size=0.2, random_state=42):
    """Loads data from .mat file, preprocesses labels, and performs train-test split."""
    
    # Load Data
    data = scipy.io.loadmat(filepath)
    signals = data['generated_signals']
    labels = data['labels']
    signal2_start_times = data['signal2_start_times']

    # One-hot encode labels
    labels = to_categorical(labels)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(signals, labels, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test, signal2_start_times

def load_doa2_data(filepaths, test_size=0.2, random_state=42):
    """
    Loads and preprocesses DoA2 data from .mat file for DoA estimation tasks.
    Returns: train/test split of signal data and their DoA2 labels.
    """
    all_signals = []
    all_labels = []

    for path in filepaths:
        data = scipy.io.loadmat(path)
        all_signals.append(data['generated_signals'])  # shape: (N, 8, T, 2)
        all_labels.append(data['doa2s'].squeeze())      # shape: (N,)

    # Combine
    signals = np.concatenate(all_signals, axis=0)
    doa2s = np.concatenate(all_labels, axis=0)

    # Encode angles to class indices
    doa_range = np.arange(-60, 61, 2)  # 61 classes
    angle_to_class = {angle: idx for idx, angle in enumerate(doa_range)}
    y_class = np.array([angle_to_class[angle] for angle in doa2s])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        signals, y_class, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, doa2s

def load_steering_vectors(filepath='steering_matrix.mat'):
    data = scipy.io.loadmat(filepath)
    steering_matrix = data['steering_matrix']     # 8 x 61
    DoAs = data['DoAs'].squeeze()                 # 61 labels
    return steering_matrix, DoAs