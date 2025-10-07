import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from tensorflow.keras.utils import to_categorical

from glrt import compute_glrt, z_score_normalize
from doa import convert_signals_to_covariance_input, evaluate_doa_predictions


def fgsm_attack(model, x, y, epsilon, norm='inf'):
    """
    Generate FGSM adversarial examples using CleverHans library.

    Parameters:
        model: Keras model
        x: Input samples (NumPy or Tensor)
        y: True labels (NumPy or Tensor, one-hot encoded)
        epsilon: Attack strength
        norm: 'inf' for L∞ or '2' for L2

    Returns:
        Adversarial examples (tf.Tensor)
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(tf.argmax(y, axis=1), dtype=tf.int64)

    norm_value = np.inf if norm == 'inf' else 2

    adv_x = fast_gradient_method(
        model_fn=model,
        x=x,
        eps=epsilon,
        norm=norm_value,
        y=y,
        clip_min=None,
        clip_max=None
    )

    return adv_x

def pgd_attack(model, x, y, epsilon, alpha=None, num_iter=10, norm='inf'):
    """
    Generate PGD adversarial examples using CleverHans library.
    Supports both class index labels and one-hot labels.

    Parameters:
        model: Keras model
        x: Input samples (NumPy or Tensor)
        y: True labels (NumPy or Tensor, one-hot or class indices)
        epsilon: Maximum perturbation (eps)
        alpha: Step size (eps_iter)
        num_iter: Number of iterations (nb_iter)
        norm: 'inf' or '2' for L-infinity / L2 norm

    Returns:
        Adversarial examples (tf.Tensor)
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # Normalize labels to integer class indices
    if isinstance(y, np.ndarray) and y.ndim > 1 and y.shape[-1] > 1:
        y = tf.convert_to_tensor(np.argmax(y, axis=1), dtype=tf.int64)
    else:
        y = tf.convert_to_tensor(y, dtype=tf.int64)

    norm_val = np.inf if norm == 'inf' else 2

    if alpha is None:
        alpha = epsilon / 10

    adv_x = projected_gradient_descent(
        model_fn=model,
        x=x,
        eps=epsilon,
        eps_iter=alpha,
        nb_iter=num_iter,
        norm=norm_val,
        y=y,
        clip_min=-1.0,
        clip_max=1.0
    )

    return adv_x

def evaluate_attack_across_epsilons(model, X_test, y_test, epsilon_values, glrt_threshold, 
                                    attack_type='fgsm', norm='inf', alpha=None, num_iter=10):
    """
    Evaluate CNN and GLRT accuracy + predictions across varying epsilon values under FGSM or PGD attack.

    Returns:
        cnn_accuracies: array of CNN accuracy per epsilon
        glrt_accuracies: array of GLRT accuracy per epsilon
        cnn_preds_all: list of CNN predictions per epsilon
        glrt_preds_all: list of GLRT predictions per epsilon
    """

    cnn_accuracies = np.zeros(len(epsilon_values))
    glrt_accuracies = np.zeros(len(epsilon_values))
    y_test_binary = np.argmax(y_test, axis=1)

    cnn_preds_all = []
    glrt_preds_all = []

    for idx, epsilon in enumerate(epsilon_values):
        print(f"[{attack_type.upper()}-{norm}] Epsilon = {epsilon:.5f}")

        # Assign alpha if PGD
        alpha_val = alpha
        if attack_type == 'pgd' and alpha is None:
            alpha_val = epsilon / 10

        # Generate adversarial examples
        if attack_type == 'fgsm':
            X_adv = fgsm_attack(model, X_test, y_test, epsilon, norm=norm)
        elif attack_type == 'pgd':
            X_adv = pgd_attack(model, X_test, y_test, epsilon, alpha=alpha_val, num_iter=num_iter, norm=norm)
        else:
            raise ValueError("Unsupported attack type. Use 'fgsm' or 'pgd'.")

        # CNN predictions
        cnn_output = model.predict(X_adv, verbose=0)
        cnn_preds = np.argmax(cnn_output, axis=1)
        cnn_acc = np.mean(cnn_preds == y_test_binary)
        cnn_accuracies[idx] = cnn_acc
        cnn_preds_all.append(cnn_preds)

        # GLRT stats
        attacked = X_adv.numpy()
        attacked_complex = attacked[..., 0] + 1j * attacked[..., 1]
        glrt_stats = np.array([
            np.max(z_score_normalize(compute_glrt(sig))) for sig in attacked_complex
        ])

        glrt_preds = (glrt_stats > glrt_threshold).astype(int)
        glrt_acc = accuracy_score(y_test_binary, glrt_preds)
        glrt_accuracies[idx] = glrt_acc
        glrt_preds_all.append(glrt_preds)

    return cnn_accuracies, glrt_accuracies, cnn_preds_all, glrt_preds_all

def evaluate_attack_across_epsilons_doa(model, X_test, y_test, steering_matrix,
                                        epsilon_values, attack_type='fgsm', norm='inf',
                                        turn_on_time=750, cnn_block_size=750,
                                        tolerance=2, alpha=None, num_iter=10):
    """
    Evaluate CNN and GLRT DoA accuracy using different block sizes and compute perturbation norms.
    Applies adversarial attack only to CNN input (covariance matrices).
    GLRT is evaluated on perturbed signal input.

    Returns:
        cnn_accuracies, glrt_accuracies, cnn_norms, glrt_norms, cnn_preds_all, glrt_preds_all
    """
    cnn_accuracies, glrt_accuracies = [], []
    cnn_preds_all, glrt_preds_all = [], []

    y_test_onehot = to_categorical(y_test, num_classes=model.output_shape[-1])

    # Precompute CNN input from clean signals
    X_test_cov = convert_signals_to_covariance_input(X_test, turn_on_time, block_size=cnn_block_size)

    for idx, epsilon in enumerate(epsilon_values):
        print(f"[{attack_type.upper()}-{norm}] Epsilon = {epsilon:.5f}")

        # Apply adversarial attack to CNN input only
        if attack_type == 'fgsm':
            X_adv_cov = fgsm_attack(model, X_test_cov, y_test_onehot, epsilon, norm=norm)
        elif attack_type == 'pgd':
            alpha_val = alpha if alpha else epsilon / 10
            X_adv_cov = pgd_attack(model, X_test_cov, y_test, epsilon, alpha=alpha_val, num_iter=num_iter, norm=norm)
        else:
            raise ValueError("Unsupported attack type")

        X_adv_cov = X_adv_cov.numpy()

        # CNN prediction on perturbed input
        cnn_output = model.predict(X_adv_cov, verbose=0)
        cnn_preds = np.argmax(cnn_output, axis=1)
        cnn_acc = np.mean(cnn_preds == y_test)
        cnn_accuracies.append(cnn_acc)
        cnn_preds_all.append(cnn_preds)

        # GLRT prediction on perturbed input
        glrt_acc, glrt_preds = evaluate_doa_predictions(
            X_adv_cov, y_test_onehot, steering_matrix,
            tolerance=tolerance
        )
        glrt_accuracies.append(glrt_acc)
        glrt_preds_all.append(glrt_preds)

    return cnn_accuracies, glrt_accuracies, cnn_preds_all, glrt_preds_all

def compute_psr_db_l2(epsilon):
    """
    Compute PSR (in dB) for L2-norm bounded perturbations.
    PSR_dB = 10 * log10(ε^2) = 20 * log10(ε)

    Args:
        epsilon (float or np.ndarray): L2 epsilon value(s)

    Returns:
        float or np.ndarray: PSR in dB
    """
    return 10 * np.log10(epsilon**2)


def compute_psr_db_linf(epsilon, num_features):
    """
    Compute PSR (in dB) for L∞-norm bounded perturbations.
    PSR_dB = 10 * log10(ε^2 * d) = 20 * log10(ε * sqrt(d))

    Args:
        epsilon (float or np.ndarray): Linf epsilon value(s)
        num_features (int): Total number of input features (d)

    Returns:
        float or np.ndarray: PSR in dB
    """
    return 10 * np.log10((epsilon**2) * num_features)