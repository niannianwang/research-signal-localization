import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import models, Sequential
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Activation, Input, Lambda, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from cnn import load_model, load_doa_model
from adversarial_attacks import fgsm_attack, pgd_attack

# Adversarial training
def adversarial_train_and_save(
    X_train, y_train,
    epsilon, norm,
    save_path,
    attack_type='pgd',
    model_file='cnn',
    load_path='../models/noattack',
    validation_split=0.1,
    epochs=20,
    batch_size=128
):
    # Load detection model
    model = load_model(model_file, load_dir=load_path)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Generate adversarial examples
    if attack_type == 'pgd':
        X_adv = pgd_attack(model, X_train, y_train, epsilon=epsilon, norm=norm)
    elif attack_type == 'fgsm':
        X_adv = fgsm_attack(model, X_train, y_train, epsilon=epsilon, norm=norm)
    else:
        raise ValueError(f"Unsupported attack_type: {attack_type}")

    # Combine clean + adversarial
    X_train_adv = np.concatenate([X_train, X_adv], axis=0)
    y_train_adv = np.concatenate([y_train] * 2, axis=0)

    # Shuffle
    idx = np.random.permutation(len(X_train_adv))
    X_train_adv = X_train_adv[idx]
    y_train_adv = y_train_adv[idx]

    # Callbacks (same as DoA version)
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # Train
    model.fit(
        X_train_adv, y_train_adv,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Save
    model.save(save_path)
    print(f"Model saved at {save_path}")
    return model


# Conduct adversarial training and save the DoA model
def adversarial_train_and_save_doa(
    X_train, y_train,
    epsilon, norm,
    save_path,
    attack_type='pgd',
    model_name='cnn_doa_cov',
    load_path='../models/noattack',
    validation_split=0.1,
    epochs=20,
    batch_size=128
):
    # Load DoA model
    model = load_doa_model(model_name, load_dir=load_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Generate adversarial examples
    if attack_type == 'pgd':
        X_adv = pgd_attack(model, X_train, y_train, epsilon=epsilon, norm=norm)
    elif attack_type == 'fgsm':
        X_adv = fgsm_attack(model, X_train, y_train, epsilon=epsilon, norm=norm)
    else:
        raise ValueError(f"Unsupported attack_type: {attack_type}")

    # Combine clean + adversarial
    X_train_adv = np.concatenate([X_train, X_adv], axis=0)
    y_train_adv = np.concatenate([y_train]*2, axis=0)

    # Shuffle
    idx = np.random.permutation(len(X_train_adv))
    X_train_adv = X_train_adv[idx]
    y_train_adv = y_train_adv[idx]

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # Train
    model.fit(
        X_train_adv, y_train_adv,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Save
    model.save(save_path)
    print(f"Model saved at {save_path}")
    return model

###############################################################################
# Distillation training

def softmax_with_temperature(T):
    return Lambda(lambda x: tf.nn.softmax(x / T))

def build_defensive_distillation_model(input_shape, num_classes, T=1.0):
    return Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1, 2)),

        Flatten(),
        Dense(128, activation='linear'), # try linear activation
        Dropout(0.3),
        Dense(num_classes),
        softmax_with_temperature(T)
    ])
    
def train_phase1_model(X_train, y_train, T=1.0, save_path=None):
    model = build_defensive_distillation_model(X_train.shape[1:], y_train.shape[1], T)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        batch_size=128,
        epochs=30,
        shuffle=True,
        callbacks=[
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=2)
        ]
    )
    if save_path:
        model.save(save_path)
        print(f"Phase 1 model saved to {save_path}")
    return model

def train_phase2_model(X_train, soft_labels, T=1.0, save_path=None):
    model = build_defensive_distillation_model(X_train.shape[1:], soft_labels.shape[1], T)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=KLDivergence(), metrics=['accuracy'])

    model.fit(
        X_train, soft_labels,
        validation_split=0.1,
        batch_size=128,
        epochs=30,
        shuffle=True,
        callbacks=[
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=2)
        ]
    )
    if save_path:
        model.save(save_path)
        print(f"Phase 2 (distilled) model saved to {save_path}")
    return model

def create_inference_model(distilled_model_path):
    student = tf.keras.models.load_model(distilled_model_path, compile=False, safe_mode=False)

    # Remove temperature softmax
    layers = student.layers[:-1]
    softmax_layer = Activation('softmax')

    model = Sequential(layers + [softmax_layer])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_defensive_distillation_model_doa(input_shape, num_classes, T=1.0):
    return Sequential([
        Input(shape=input_shape),
        Conv2D(32, (2, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(1, 2)),

        Conv2D(64, (2, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(1, 2)),

        Flatten(),
        Dense(128, activation='linear'),
        Dropout(0.3),
        Dense(num_classes),
        softmax_with_temperature(T)
    ])

def train_phase1_model_doa(X_train, y_train, T, save_path=None):
    model = build_defensive_distillation_model_doa(X_train.shape[1:], y_train.shape[1], T)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        batch_size=128,
        epochs=30, # before was 20
        shuffle=True,
        callbacks=[
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=2)
        ]
    )
    if save_path:
        model.save(save_path)
        print(f"Phase 1 model saved to {save_path}")
    return model

def train_phase2_model_doa(X_train, soft_labels, T, save_path=None):
    model = build_defensive_distillation_model_doa(X_train.shape[1:], soft_labels.shape[1], T)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=KLDivergence(), metrics=['accuracy'])

    model.fit(
        X_train, soft_labels,
        validation_split=0.1,
        batch_size=128,
        epochs=30, # before was 20
        shuffle=True,
        callbacks=[
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=2)
        ]
    )
    if save_path:
        model.save(save_path)
        print(f"Phase 2 (distilled) model saved to {save_path}")
    return model

def create_inference_model_doa(distilled_model_path):
    student = tf.keras.models.load_model(distilled_model_path, compile=False, safe_mode=False)

    # Remove softmax-with-T
    layers = student.layers[:-1]
    softmax_layer = Activation('softmax')

    model = Sequential(layers + [softmax_layer])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

###############################################################################
# DAE

def train_denoising_autoencoder(
    X_train, y_train,
    epsilon, norm,
    save_path,
    attack_type='fgsm',
    model_name='cnn',
    load_path='../models/noattack',
    validation_split=0.1,
    epochs=20,
    batch_size=128
):
    """
    Train a convolutional denoising autoencoder to reconstruct clean input from adversarial input,
    then stack it with a pre-trained classifier for robustness evaluation.

    Parameters:
        X_train: np.ndarray - clean input (N, 8, 500, 2)
        y_train: np.ndarray - one-hot labels (N, 2)
        epsilon: float - attack strength
        norm: str - 'inf' or 'l2'
        save_path: str - where to save full AE+classifier model
        attack_type: str - 'fgsm' or 'pgd'
        model_name: str - classifier to load (no `.keras`)
        load_path: str - path to classifier directory
        validation_split: float - validation ratio
        epochs: int - training epochs
        batch_size: int - training batch size

    Returns:
        combined_model: keras.Model - autoencoder + classifier pipeline
    """
    input_shape = X_train.shape[1:]

    # Load model for attack generation
    model = load_model(model_name, load_dir=load_path)
    if attack_type == 'fgsm':
        X_noisy = fgsm_attack(model, X_train, y_train, epsilon, norm)
    elif attack_type == 'pgd':
        X_noisy = pgd_attack(model, X_train, y_train, epsilon=epsilon, norm=norm)
    else:
        raise ValueError("Unsupported attack type")

    if isinstance(X_noisy, tf.Tensor):
        X_noisy = X_noisy.numpy()

    # Combine clean and noisy data for denoising training
    X_train_ae = np.concatenate([X_noisy, X_train])
    y_train_ae = np.concatenate([X_train, X_train])

    # Define convolutional DAE architecture
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='linear', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='linear', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), activation='linear', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='linear', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    output_layer = Conv2D(2, (3, 3), activation='linear', padding='same')(x)

    autoencoder = models.Model(input_layer, output_layer, name='dense_autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(
        X_train_ae, y_train_ae,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ],
        verbose=1
    )

    # Load and freeze detection CNN
    classifier = load_model(model_name, load_dir=load_path)
    classifier.trainable = False

    # Stack AE + CNN
    combined_input = Input(shape=input_shape)
    x_reconstructed = autoencoder(combined_input)
    predictions = classifier(x_reconstructed)
    combined_model = tf.keras.Model(combined_input, predictions, name='dae_classifier')

    combined_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    combined_model.save(save_path)
    print(f"Combined AE + Classifier saved to {save_path}")
    return combined_model

def train_denoising_autoencoder_doa(
    X_train, y_train,
    epsilon, norm,
    save_path,
    attack_type='fgsm',
    model_name='cnn_doa_cov',
    load_path='../models/noattack',
    validation_split=0.1,
    epochs=20,
    batch_size=128
):
    """
    Train a denoising autoencoder to reconstruct clean input from adversarial input,
    then stack it with a pre-trained classifier for robustness evaluation.

    Parameters:
        X_train: np.ndarray - clean covariance input (N, H, W, C)
        y_train: np.ndarray - one-hot labels (N, num_classes)
        epsilon: float - attack strength
        norm: str - 'inf' or 'l2'
        save_path: str - where to save full AE+classifier model
        attack_type: str - 'fgsm' or 'pgd'
        model_name: str - name of classifier to load
        load_path: str - path to pre-trained classifier
        validation_split: float - portion of AE data to use for validation
        epochs: int - number of training epochs
        batch_size: int - training batch size

    Returns:
        combined_model: keras.Model - autoencoder + classifier pipeline
    """
    # Flatten input from (8, 16, 2) to (256,)
    X_flat = X_train.reshape((X_train.shape[0], -1))
    input_dim = X_flat.shape[1]

    # Generate adversarial noisy inputs
    model = load_doa_model(model_name, load_dir=load_path)
    if attack_type == 'fgsm':
        X_noisy = fgsm_attack(model, X_train, y_train, epsilon, norm)
    elif attack_type == 'pgd':
        X_noisy = pgd_attack(model, X_train, y_train, epsilon=epsilon, norm=norm)
    else:
        raise ValueError("Unsupported attack type")

    if isinstance(X_noisy, tf.Tensor):
        X_noisy = X_noisy.numpy()
    X_noisy_flat = X_noisy.reshape((X_noisy.shape[0], -1))
    X_train_ae = np.concatenate([X_noisy_flat, X_flat])
    y_train_ae = np.concatenate([X_flat, X_flat])

    # Build fully connected
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(512, activation='linear')(input_layer)
    encoded = Dense(256, activation='linear')(encoded)
    bottleneck = Dense(128, activation='linear')(encoded)
    decoded = Dense(256, activation='linear')(bottleneck)
    decoded = Dense(512, activation='linear')(decoded)
    output_layer = Dense(input_dim)(decoded)

    autoencoder = models.Model(input_layer, output_layer, name='dense_autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(
        X_train_ae, y_train_ae,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ],
        verbose=1
    )

    # Load and freeze classifier
    classifier = tf.keras.models.load_model(f"{load_path}/{model_name}.keras")
    classifier.trainable = False

    # Stack AE + classifier with reshape
    combined_input = Input(shape=(input_dim,))
    x_reconstructed = autoencoder(combined_input)
    x_reshaped = tf.keras.layers.Reshape((8, 16, 2))(x_reconstructed)
    predictions = classifier(x_reshaped)
    combined_model = tf.keras.Model(combined_input, predictions, name='dae_classifier')

    combined_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    combined_model.save(save_path)
    print(f"Combined AE + Classifier saved to {save_path}")
    return combined_model

###############################################################################

# Evaluate CNN model under adversarial attacks across multiple epsilon values
def evaluate_cnn_across_epsilons(model, X_test, y_test, epsilon_values, 
                                  attack_type='fgsm', norm='inf', alpha=None, num_iter=10):
    """
    Evaluate CNN model under FGSM or PGD attack across a range of epsilon values.

    Parameters:
        model: Trained CNN model
        X_test: Test data (NumPy array)
        y_test: One-hot encoded labels (NumPy array)
        epsilon_values: Array of epsilon values
        attack_type: 'fgsm' or 'pgd'
        norm: Norm to use ('inf' or '2')
        alpha: Step size for PGD (if None, will default to epsilon/10)
        num_iter: Number of PGD iterations

    Returns:
        cnn_accuracies: List of accuracy per epsilon
        cnn_preds_all: List of prediction arrays per epsilon
    """
    cnn_accuracies = []
    cnn_preds_all = []
    y_true = np.argmax(y_test, axis=1)
    
    # Reshape input for DAE
    if model.input_shape[-1] == 256 and X_test.ndim == 4:
        X_test = X_test.reshape((X_test.shape[0], -1))

    for idx, epsilon in enumerate(epsilon_values):
        print(f"[{attack_type.upper()}-{norm}] Epsilon = {epsilon:.5f}")

        # Determine alpha if using PGD
        alpha_val = alpha
        if attack_type == 'pgd' and alpha is None:
            alpha_val = epsilon / 10

        # Generate adversarial examples
        if attack_type == 'fgsm':
            if epsilon == 0.0:
                X_adv = X_test.copy()
            else:
                X_adv = fgsm_attack(model, X_test, y_test, epsilon=epsilon, norm=norm)
        elif attack_type == 'pgd':
            X_adv = pgd_attack(model, X_test, y_test, epsilon=epsilon, alpha=alpha_val, num_iter=num_iter, norm=norm)
        else:
            raise ValueError("Unsupported attack type. Use 'fgsm' or 'pgd'.")

        diff = np.mean(np.abs(X_adv - X_test))
        print(f"Mean absolute perturbation at epsilon={epsilon:.5f}: {diff:.6f}")

        # CNN predictions and accuracy
        cnn_output = model.predict(X_adv, verbose=0)
        cnn_preds = np.argmax(cnn_output, axis=1)
        cnn_acc = np.mean(cnn_preds == y_true)
        cnn_accuracies.append(cnn_acc)
        cnn_preds_all.append(cnn_preds)

    return cnn_accuracies, cnn_preds_all