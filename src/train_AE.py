import os
import sys
import argparse
import numpy as np
import json
import mlflow
import tempfile
import mlflow.tensorflow
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import pairwise
from keras_flops import get_flops

# Корень проекта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import build_autoencoder
from dataset import folder_extract, gestures, train_test_split, apply_window
from config import *
from utils import set_seed, evaluate_metrics

set_seed(seed=GLOBAL_SEED)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.get_logger().setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_autoencoder(model: tf.keras.Model, epochs: int, X_train: np.ndarray, X_valid: np.ndarray, batch_size: int = 64, 
                      lr: float = 1e-3, decay_rate: float = 0.9, save_path: str = None, patience: int = 10) -> tf.keras.callbacks.History:
    """
    Обучение автоэнкодера с ранней остановкой.

    Args:
        model (tf.keras.Model): Автоэнкодер (Keras модель).
        epochs (int): Количество эпох.
        X_train (np.ndarray): Тренировочные данные (X -> X).
        X_valid (np.ndarray): Валидационные данные (X -> X).
        batch_size (int, optional): Размер батча.
        lr (float, optional): Начальная скорость обучения.
        decay_rate (float, optional): Темп экспоненциального затухания learning rate.
        save_path (str, optional): Путь для сохранения лучших весов.
        patience (int, optional): Число эпох без улучшения до остановки.

    Returns:
        tf.keras.callbacks.History: История обучения.
    """
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience, verbose=1)]
    
    if save_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                               monitor='val_loss',
                                               save_best_only=True,
                                               save_weights_only=True,
                                               mode='min',
                                               verbose=1)
        )

    steps = (len(X_train) / batch_size) * 1.5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=steps,
        decay_rate=decay_rate
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')

    history = model.fit(
        X_train, X_train,
        validation_data=(X_valid, X_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history

def main():
    mlflow.tensorflow.autolog()
    mlflow.set_experiment(f"AE trials")

    # Сырые сигналы и метки
    emg, label = folder_extract(FOLDER_PATH, exercises=EXERCISES, myo_pref=MYO_PREF)
    all_g = gestures(emg, label, targets=GESTURE_INDEXES_MAIN)

    # Train/Test split по жестам
    train_g, test_g = train_test_split(all_g, split_size=0.2, rand_seed=GLOBAL_SEED)

    # Преобразование в окна: [N, channels, window]
    X_train_raw, y_train = apply_window(train_g, window=WINDOW_SIZE, step=8)
    X_test_raw,  y_test  = apply_window(test_g,  window=WINDOW_SIZE, step=8)

    input_shape = (WINDOW_SIZE, len(CHANNELS), 1)

    os.makedirs('normalization_values', exist_ok=True)

    means = X_train_raw.mean(axis=(0, 2))       # (channels,)
    stds  = X_train_raw.std(axis=(0, 2)) + 1e-8
    def standardize(X):
        return (X - means[None,:,None]) / stds[None,:,None]
    
    X_train = standardize(X_train_raw)
    X_test= standardize(X_test_raw)
    
    def prepare(X):
        Xt = np.transpose(X, (0, 2, 1))   # [N, window, channels]
        sel = Xt[..., CHANNELS]           # отбор каналов
        return sel[..., np.newaxis].astype(np.float32)

    X_train = prepare(X_train)
    X_test = prepare(X_test)

    params = {'mean': means.tolist(), 'std':  stds.tolist()}
    norm_file = f'normalization_values/AE_{WINDOW_SIZE}.json'
    with open(norm_file, 'w') as f:
        json.dump(params, f)

    autoencoder, encoder = build_autoencoder()
    train_autoencoder(autoencoder, 10, X_train=X_train, X_valid=X_test, batch_size=2**9, lr=INIT_LR_AE)
    print(autoencoder.predict(X_test))
    print(autoencoder.predict(X_test).shape)

    X_test_reconstructed = autoencoder.predict(X_test)
    N = X_test_reconstructed.shape[0]  # число примеров
    X_test_flat = X_test.reshape(N, -1)
    X_test_recon_flat = X_test_reconstructed.reshape(N, -1)

    sim = pairwise.cosine_similarity(X_test_flat, X_test_recon_flat).diagonal().mean()

    print(encoder.predict(X_test).shape)
    print(autoencoder.predict(X_test).shape)



if __name__ == '__main__':
    main()

