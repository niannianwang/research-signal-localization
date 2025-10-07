import numpy as np
import matplotlib.pyplot as plt

def compute_glrt(signal, block_size=10):
    """
    Compute the GLRT statistics for one signal.
    
    Parameters:
        signal (numpy array): Input signal of shape (array_size, N) where array_size is the number of sensors
                              and N is the number of time samples.
        block_size (int): The size of the block for covariance matrix computation.
        
    Returns:
        glrt_statistics (numpy array): Array of GLRT statistics.
    """
    array_size, N = signal.shape[0], signal.shape[1]
    num_iterations = N - block_size + 1
    epsilon = 1e-6 # Regularization parameter

    # Initialize covariance matrices
    cov_matrices = np.zeros((array_size, array_size, num_iterations), dtype=np.complex128)

    # Compute covariance matrices for each block shifted by one unit of time in each iteration
    for i in range(num_iterations):
        curr_block = signal[:, i:i + block_size]
        cov_matrices[:, :, i] = np.cov(curr_block)

    # Compute GLRT statistics
    glrt_statistics = np.zeros(num_iterations - 1)
    for i in range(num_iterations - 1):
        # Covariance of block i
        R1 = cov_matrices[:, :, i]
        R1 += epsilon * np.eye(R1.shape[0])
        # Covariance of block i + 1
        R2 = cov_matrices[:, :, i + 1]
        R2 += epsilon * np.eye(R2.shape[0])
        # GLRT statistic
        glrt_statistics[i] = np.abs(np.trace(np.linalg.inv(R1) @ R2))

    return glrt_statistics

def z_score_normalize(glrt_stat):
    """
    Normalize GLRT statistics using Z-score normalization.
    
    Parameters:
        glrt_stat (numpy array): Array of GLRT statistic values.
    
    Returns:
        normalized_glrt (numpy array): Z-score normalized GLRT statistic values.
    """
    mean = np.mean(glrt_stat)
    std = np.std(glrt_stat) + 1e-6
    return (glrt_stat - mean) / std

def find_glrt_threshold(X_train, y_train, block_size=10, epsilon=1e-6):
    """
    Find a GLRT threshold by computing the max GLRT value among samples without Signal 2.
    
    Parameters:
        X_train_combined (numpy array): Training set of complex signals with shape (num_samples, array_size, N).
        y_train (numpy array): Labels for the training set (one-hot encoded or binary).
        block_size (int): The size of the block for covariance computation.
        epsilon (float): Regularization parameter to ensure invertibility of covariance matrices.
        
    Returns:
        threshold (float): Maximum GLRT value from samples without Signal 2.
        glrt_values_signal2 (list): List of GLRT statistic values for samples with Signal 2.
    """
    
    # Combine real and imaginary parts of the signals
    X_train_combined = X_train[:, :, :, 0] + 1j * X_train[:, :, :, 1]
    
    glrt_values_no_signal2 = []  # Stores GLRT values for samples without Signal 2
    
    for i in range(X_train_combined.shape[0]):
        if y_train[i, 1] == 0:
            train_signal = X_train_combined[i, :, :]
            glrt_stat = compute_glrt(train_signal, block_size=block_size)  # Compute GLRT statistic
            glrt_stat = z_score_normalize(glrt_stat)  # Normalize GLRT statistic
            max_glrt = max(glrt_stat)  # Get max GLRT statistic for this sample
            glrt_values_no_signal2.append(max_glrt)
        
    # Set threshold as the max GLRT value from samples without Signal 2
    threshold = np.percentile(glrt_values_no_signal2, 95)
    
    # Plot histogram of GLRT values
    plt.hist(glrt_values_no_signal2, bins=50, alpha=0.7, color='blue', label='No Signal 2')
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.xlabel('GLRT Statistic')
    plt.ylabel('Frequency')
    plt.title('Distribution of GLRT Statistics')
    plt.legend()
    plt.show()
    
    return threshold, glrt_values_no_signal2

def evaluate_glrt(X_test, y_test, threshold, block_size=10):
    """
    Evaluate GLRT performance on a test set with z-score normalization.

    Parameters:
        X_test (numpy array): Test set with shape (num_samples, sensors, time, 2)
        y_test (numpy array): One-hot encoded labels
        threshold (float): GLRT detection threshold (e.g. 95th percentile)
        block_size (int): Block size for GLRT computation

    Returns:
        accuracy (float): GLRT classification accuracy
        predictions (np.array): Binary predictions (0 or 1)
    """
    # Combine real and imaginary into complex-valued signal
    X_test_combined = X_test[..., 0] + 1j * X_test[..., 1]

    predictions = []
    for i in range(X_test_combined.shape[0]):
        glrt_stat = compute_glrt(X_test_combined[i], block_size=block_size)
        glrt_stat = z_score_normalize(glrt_stat)  # Normalize test stat
        predictions.append(1 if max(glrt_stat) > threshold else 0)

    predictions = np.array(predictions)
    accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
    
    return accuracy, predictions