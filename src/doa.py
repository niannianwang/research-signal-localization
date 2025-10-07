import numpy as np

def compute_covariance(signal_block):
    """
    Compute sample covariance matrix of shape (8, 8)

    Parameters:
        signal_block: np.ndarray of shape (8, T), complex

    Returns:
        covariance: np.ndarray of shape (8, 8)
    """
    return signal_block @ signal_block.conj().T / signal_block.shape[1]

def extract_covariances_blocks(full_signal, turn_on_time, block_size=10):
    """
    Extract R_old and R_new using sliding block windows around the signal2 onset.

    Parameters:
        full_signal: np.ndarray of shape (8, T, 2)
        turn_on_time: int — index where signal 2 starts
        block_size: int — number of time steps per block

    Returns:
        R_old, R_new: np.ndarrays of shape (8, 8)
    """
    signal_complex = full_signal[..., 0] + 1j * full_signal[..., 1]

    # Define exact blocks
    R_old_block = signal_complex[:, turn_on_time - block_size:turn_on_time]
    R_new_block = signal_complex[:, turn_on_time : turn_on_time + block_size] # First full signal2 window
    # R_new_block = signal_complex[:, turn_on_time + 1 : turn_on_time + 1 + block_size] # One after fist full signal2 window

    R_old = compute_covariance(R_old_block)
    R_new = compute_covariance(R_new_block)
    
    R_old /= np.linalg.norm(R_old, ord='fro') + 1e-10
    R_new /= np.linalg.norm(R_new, ord='fro') + 1e-10

    return R_old, R_new

def compute_max_eigenvector(R_old, R_new, epsilon=1e-6):
    """
    Compute R_old^-1 * R_new (via solve) and get dominant eigenvector.

    Parameters:
        R_old, R_new: np.ndarray (8x8) covariance matrices

    Returns:
        v_max: dominant eigenvector (8,)
    """
    R_old += epsilon * np.eye(R_old.shape[0])
    
    R_old_inv = np.linalg.inv(R_old)
    R_transform = R_old_inv @ R_new
    eigenvalues, eigenvectors = np.linalg.eig(R_transform)
    max_idx = np.argmax(np.abs(eigenvalues))
    return eigenvectors[:, max_idx]

def predict_doa_from_eigenvector(v_max, steering_matrix):
    """
    Predict DoA index by computing correlation between v_max and all steering vectors.

    Parameters:
        v_max: np.ndarray of shape (8,), complex eigenvector
        steering_matrix: np.ndarray of shape (8, num_angles)

    Returns:
        predicted_index: int — index in the DoA class list with highest similarity
    """
    correlations = np.abs(steering_matrix.conj().T @ v_max)  # (num_angles,)
    return np.argmax(correlations)

def predict_doa(sample, steering_matrix, turn_on_time=500, block_size=10):
    """
    Full GLRT DoA prediction pipeline for a single signal sample.

    Parameters:
        sample: np.ndarray of shape (8, T, 2)
        steering_matrix: np.ndarray of shape (8, num_angles)

    Returns:
        predicted_index: int — index in steering_matrix with max match
    """
    R_old, R_new = extract_covariances_blocks(sample, turn_on_time, block_size) # slice
    v_max = compute_max_eigenvector(R_old, R_new)
    return predict_doa_from_eigenvector(v_max, steering_matrix)

def evaluate_doa_predictions_from_raw(X, y_true, steering_matrix, turn_on_time=750, block_size=750, tolerance=2):
    """
    Run DoA prediction on all samples and evaluate accuracy with index tolerance.

    Parameters:
        X: np.ndarray of shape (N, 8, T, 2)
        y_true: np.ndarray of shape (N,)
        steering_matrix: np.ndarray of shape (8, num_angles)
        tolerance: int — tolerance in index space (e.g., ±2)

    Returns:
        accuracy: float (0.0 to 1.0)
        predictions: np.ndarray of shape (N,)
    """
    # Check if y_true is one-hot encoded and convert to class indices
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    predictions = []
    for sample in X:
        pred = predict_doa(sample, steering_matrix, turn_on_time, block_size)
        predictions.append(pred)

    predictions = np.array(predictions)
    correct = np.abs(predictions - y_true) <= tolerance
    accuracy = np.mean(correct)

    return accuracy, predictions

def evaluate_doa_predictions(X, y_true, steering_matrix, tolerance=2):
    """
    Evaluate DoA predictions from precomputed covariance matrices (R_old and R_new).

    Parameters:
        X: np.ndarray of shape (N, 8, 16, 2) — real+imag R_old || R_new
        y_true: np.ndarray of shape (N,) or one-hot
        steering_matrix: np.ndarray of shape (8, num_angles)
        tolerance: int — tolerance in index space (e.g., ±2)

    Returns:
        accuracy: float
        predictions: np.ndarray of shape (N,)
    """
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    predictions = []
    for sample in X:
        cov_complex = sample[..., 0] + 1j * sample[..., 1]  # shape (8, 16)
        R_old_cov = cov_complex[:, :8]  # shape (8, 8)
        R_new_cov = cov_complex[:, 8:]  # shape (8, 8)

        v_max = compute_max_eigenvector(R_old_cov, R_new_cov)
        pred = predict_doa_from_eigenvector(v_max, steering_matrix)
        predictions.append(pred)

    predictions = np.array(predictions)
    correct = np.abs(predictions - y_true) <= tolerance
    accuracy = np.mean(correct)

    return accuracy, predictions

def convert_signals_to_covariance_input(X, turn_on_time=750, block_size=10):
    """
    Converts full signal dataset to (N, 8, 16, 2) CNN-ready format using R_old and R_new.

    Parameters:
        X: np.ndarray, shape (N, 8, T, 2)
        turn_on_time: int, signal2 onset
        block_size: int, time window for each covariance block

    Returns:
        np.ndarray: shape (N, 8, 16, 2)
    """
    N = X.shape[0]
    cov_inputs = np.zeros((N, 8, 16, 2))  # real+imag from R_old (8x8) and R_new (8x8)

    for i in range(N):
        R_old, R_new = extract_covariances_blocks(X[i], turn_on_time, block_size)
        
        R_old /= np.linalg.norm(R_old, ord='fro') + 1e-10
        R_new /= np.linalg.norm(R_new, ord='fro') + 1e-10

        R_concat = np.concatenate([R_old, R_new], axis=1)
        cov_inputs[i, ..., 0] = R_concat.real
        cov_inputs[i, ..., 1] = R_concat.imag

    return cov_inputs