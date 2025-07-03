import os
import sys
import argparse
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from keras_flops import get_flops
from models import build_base_model
from dataset import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
from config import *
from utils import *

set_seed(seed=GLOBAL_SEED)    # Для воспроизводимости 
print(tf.config.list_physical_devices('GPU'))

tf.get_logger().setLevel('INFO')

def parse_args():
    parser = argparse.ArgumentParser(description='Обучение модели классификации жестов')

    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE, help='Размер окна')

    return parser.parse_args()

def train_model(model: tf.keras.Sequential, epochs: int, X_train, y_train, X_valid, y_valid, batch_size: int=BATCH_SIZE, lr: float=INIT_LR,
                decay_rate: float=0.9, save_path: str=SAVE_PATH, patience: int=PATIENCE):
    """Функция для обучения модели.

    Args:
        model (tf.keras.Sequential): Класс модели
        epochs (int): Количество эпох.
        X_train (_type_): Тренировочные признаки
        y_train (_type_): Тренировочные метки жестов
        X_valid (_type_): Валидационные признаки
        y_valid (_type_): Валидационные метки жестов
        batch_size (int, optional): размер батча. Defaults to BATCH_SIZE.
        lr (float, optional): Начальная скорость обучения. Defaults to INIT_LR.
        decay_rate (float, optional): _description_. Defaults to 0.9.    
        save_path (str, optional): _description_. Defaults to SAVE_PATH.
        patience (int, optional): Количество эпох для ранней остановки. Defaults to PATIENCE.

    Returns:
        _type_: _description_
    """
    callback_lists = []

    # NOTE: Callback для сохранения весов модели после каждой эпохи, если val_loss уменьшился
    if save_path != None:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_freq='epoch',
                                                        save_best_only=True, mode='min', save_weights_only=True)
        callback_lists.append(checkpoint)

    # NOTE: Callback для ранней остановки
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    callback_lists.append(early)

    # NOTE: Scheduler с экспоненциальным уменьшением скорости обучения 
    decay_steps = (len(X_train) / batch_size) * 1.5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),   
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
        )
    
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                        validation_data=(X_valid, y_valid), callbacks=callback_lists)

    return history
    

def main():
    args = parse_args()

    mlflow.tensorflow.autolog()

    # NOTE: Извлечение всех данных из .mat файлов 
    emg, label = folder_extract(FOLDER_PATH, exercises=EXERCISES, myo_pref=MYO_PREF)

    # Извлечение жестов из GESTURE_INDEXES
    emg = standarization(emg, STD_MEAN_PATH)
    gest = gestures(emg, label, targets=GESTURE_INDEXES_MAIN)
    print(f'Выбранные жесты: {gest.keys()}')

    # NOTE: Разделение данных на выборки
    train_gestures, tmp_gest = train_test_split(gest, split_size=TEST_SIZE, rand_seed=GLOBAL_SEED)
    valid_gestures, test_gestures = train_test_split(tmp_gest, split_size=VALID_SIZE, rand_seed=GLOBAL_SEED)
    # print(len(train_gestures[26])) 
    X_train, y_train = apply_window(train_gestures, window=args.window_size, step=STEP_SIZE)
    X_valid, y_valid = apply_window(valid_gestures, window=args.window_size, step=STEP_SIZE)
    X_test, y_test = apply_window(test_gestures, window=args.window_size, step=STEP_SIZE)    # Вносить данные о выборках в mlflow 

    X_train = np.transpose(X_train.reshape(-1, 8, args.window_size, 1), (0, 2, 1, 3))[:, :, CHANNELS, :].astype(np.float32)
    X_valid = np.transpose(X_valid.reshape(-1, 8, args.window_size, 1), (0, 2, 1, 3))[:, :, CHANNELS, :].astype(np.float32)
    X_test = np.transpose(X_test.reshape(-1, 8, args.window_size, 1), (0, 2, 1, 3))[:, :, CHANNELS, :].astype(np.float32)

    input_shape = (args.window_size, INPUT_SHAPE_BASE[1], INPUT_SHAPE_BASE[2])
    model = build_base_model(input_shape=input_shape, filters=FILTERS_BASE, kernel_size=KERNEL_SIZE_BASE, 
                             pool_size=POOL_SIZE_BASE, p_dropout=P_DROPOUT_BASE, num_classes=NUM_CLASSES) 
    mflops = get_flops(model, batch_size=1) / 1e6

    mlflow.set_experiment("Gesture Classification Window Study")
    with mlflow.start_run():
        history = train_model(model=model, epochs=EPOCHS, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(test_loss)
        print(test_accuracy)

        # model.save_weights("checkpoints/model_weights.h5")

        mlflow.log_param("window_size", args.window_size)
        mlflow.log_param("gesture_indexes", GESTURE_INDEXES_MAIN)
        mlflow.log_param("MFLOPs", mflops)
        mlflow.log_metric("test_accuracy", test_accuracy)


if __name__ == "__main__":
    main()