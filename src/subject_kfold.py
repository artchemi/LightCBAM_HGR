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
from keras_flops import get_flops

# Корень проекта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import build_base_model, build_SAM_model, build_CAM_model_1D
from dataset import folder_extract_subject, gestures, train_test_split, apply_window
from config import *
from utils import set_seed, evaluate_metrics


set_seed(seed=GLOBAL_SEED)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.get_logger().setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--window_size', type=int, default=WINDOW_SIZE)
    p.add_argument('--model', type=str, default='base', choices=['base', 'SAM'])
    p.add_argument('--channels', type=str, default='full', choices=['full', 'reduced'])
    p.add_argument('--step_size', type=int, default=STEP_SIZE)    # Расстояние между окнами
    return p.parse_args()


def train_model(model: tf.keras.Sequential, epochs: int, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray, 
                batch_size: int=BATCH_SIZE, lr: float=INIT_LR, decay_rate: float=0.9, save_path=None, patience=PATIENCE) -> None:
    """Обучение модели с ранней остановкой.

    Args:
        model (tf.keras.Sequential): Модель TF.
        epochs (int): Количество эпох.
        X_train (np.ndarray): Тренировочные окна.
        y_train (np.ndarray): Тренировочные метки жестов.
        y_valid (np.ndarray): Валидационные окна.
        X_valid (np.ndarray): Валидационные метки жестов.
        batch_size (int, optional): Размер батча. Defaults to BATCH_SIZE.
        lr (float, optional): Скорость обучения. Defaults to INIT_LR.
        decay_rate (float, optional): Темп уменьшения скорости обучения. Defaults to 0.9.
        save_path (_type_, optional): Путь для сохранения весов. Defaults to None.
        patience (_type_, optional): Порог эпох для ранней остановки. Defaults to PATIENCE.

    Returns:
        _type_: None
    """
    callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', mode='min', patience=patience)]
    if save_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, 
                                               save_weights_only=True, mode='min', verbose=1)
                                               )

    steps = (len(X_train) / batch_size) * 1.5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=steps, decay_rate=decay_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, 
                     batch_size=batch_size, callbacks=callbacks, verbose=1)


def main():
    args = parse_args()
    mlflow.tensorflow.autolog()
    mlflow.set_experiment(f"Win{args.window_size}|{args.model}|{args.channels}|subjects")

    subjects = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11']

    for subject in subjects:    # Перебор всех субъектов    
        subjects_copy = subjects.copy()
        subjects_copy.remove(subject)    # Субъекты для тренировочной выборки 

        emg_train, label_train = folder_extract_subject(root_dir=FOLDER_PATH, exercises=EXERCISES, subjects=subjects_copy)    # Сырые сигналы
        emg_test, label_test = folder_extract_subject(root_dir=FOLDER_PATH, exercises=EXERCISES, subjects=[subject])  

        all_gestures_train = gestures(emg_train, label_train, targets=GESTURE_INDEXES_MAIN)    
        all_gestures_test = gestures(emg_test, label_test, targets=GESTURE_INDEXES_MAIN)

        train_gestures, _ = train_test_split(all_gestures_train, split_size=0.0, rand_seed=GLOBAL_SEED)    # Разбиение на выборки 
        _, test_gestures = train_test_split(all_gestures_test, split_size=1.0, rand_seed=GLOBAL_SEED)

        X_train_raw, y_train = apply_window(train_gestures, window=args.window_size, step=STEP_SIZE * 10)    # Разбиение на окна
        X_test_raw, y_test = apply_window(test_gestures, window=args.window_size, step=STEP_SIZE * 10)       #? Какой ставить размер окна

        channels = CHANNELS if args.channels == 'reduced' else list(range(8))    # Активные каналы
        input_shape = (args.window_size, len(channels), 1)                       # Размерность входного окна

        means = X_train_raw.mean(axis=(0, 2))          # Среднее
        stds  = X_train_raw.std(axis=(0, 2)) + 1e-8    # Стандартное отклонение

        #* Предобработка 
        def standardize(X):
            return (X - means[None,:,None]) / stds[None,:,None]
        
        X_train = standardize(X_train_raw)
        X_test = standardize(X_test_raw)

        def prepare(X):
            Xt = np.transpose(X, (0, 2, 1))   # [N, window, channels]
            sel = Xt[..., channels]           # отбор каналов
            return sel[..., np.newaxis].astype(np.float32)
        
        X_train = prepare(X_train)
        X_test = prepare(X_test)

        #* Выбор модели
        if args.model == 'base':
            model = build_base_model(input_shape, FILTERS_BASE, KERNEL_SIZE_BASE, POOL_SIZE_BASE, P_DROPOUT_BASE, NUM_CLASSES)
            lr = INIT_LR * 1e-2
        elif args.model == 'SAM':
            model = build_SAM_model(input_shape, FILTERS_BASE, KERNEL_SIZE_BASE, POOL_SIZE_BASE, P_DROPOUT_BASE, NUM_CLASSES)
            lr = 1e-2    # Для модели с механизмом внимания надо выбирать скорость обучения ниже бейзлайна 
        elif args.model == 'CAM':
            model = build_CAM_model_1D(input_shape, return_attention_mask=True)
            lr = 1e-2 
        else: 
            sys.exit(0)

        save_w = SAVE_PATH + f'_subject{subject}_{args.window_size}_{args.model}_{args.channels}.h5'

        with mlflow.start_run(run_name=f'subject {subject}'):
            train_model(model, EPOCHS, X_train=X_train, y_train=y_train, X_valid=X_test, y_valid=y_test, batch_size=BATCH_SIZE, lr=lr, save_path=save_w)
            model.load_weights(save_w)

            _, test_acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
            f1, report_dict, cm_df = evaluate_metrics(model, X_test, y_test)    #! Надо переписать тестирование и добавить новую метрику

            #* Логирование результатов
            mlflow.log_metric('test_accuracy', float(test_acc))
            mlflow.log_metric('test_f1', float(f1))

            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                cm_df.to_csv(tmp.name)
                mlflow.log_artifact(tmp.name, 'confusion_matrix')

            mlflow.log_dict(report_dict, 'classification_report_test.json')
            mlflow.log_param('gesture_indexes', GESTURE_INDEXES_MAIN)
            mlflow.log_param('channels', channels)

        tf.keras.backend.clear_session()

if __name__ == '__main__':
    main()
