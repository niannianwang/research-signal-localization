import os
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def CNN(X_train, y_train, model_name='cnn_model', save_dir='models'):
    """
    Create and train a CNN model for signal classification.
    
    Parameters:
        X_train (numpy array): Training data
        y_train (numpy array): Training labels
        model_name (str): Name of the model for saving
        save_dir (str): Directory to save the trained model
        
    Returns:
        model: Trained model
        history: Training history
    """
    model = models.Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(8, 500, 2)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(1, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the model
    model_path = os.path.join(save_dir, f"{model_name}.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, history

def load_model(model_name='cnn_model', load_dir='models'):
    """
    Load a previously saved CNN model.
    
    Parameters:
        model_name (str): Name of the model to load
        load_dir (str): Directory where the model is saved
        
    Returns:
        model: Loaded model
    """
    model_path = os.path.join(load_dir, f"{model_name}.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    model = models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

def CNN_DOA(X_train, y_train, model_name='cnn_doa_cov', save_dir='models/noattack', num_classes=61):
    """
    Create and train a CNN model for DoA classification (20 bins).

    Parameters:
        X_train: np.ndarray, shape (N, 8, 16, 2) — Covariance input (real + imag)
        y_train: np.ndarray, shape (N, num_classes) — One-hot encoded DoA labels
        model_name: Name for saving model
        save_dir: Directory to save models
        num_bins: Number of bins to group DoAs into (default 20)

    Returns:
        model: Trained model
        history: Training history
    """
    # Build CNN model
    model = models.Sequential([
        Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(8, 16, 2)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(1, 2)),

        Conv2D(64, kernel_size=(2, 2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(1, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    best_model_path = os.path.join(save_dir, f"{model_name}.keras")
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint, lr_scheduler],
    )

    print(f"Best checkpoint model saved to {best_model_path}")

    return model, history

def load_doa_model(model_name='cnn_doa_cov', load_dir='models/noattack'):
    """
    Load a previously saved CNN model for DoA classification.
    
    Parameters:
        model_name (str): Name of the model to load
        load_dir (str): Directory where the model is saved
        
    Returns:
        model: Loaded model
    """
    model_path = os.path.join(load_dir, f"{model_name}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    model = models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model