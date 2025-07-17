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
from models import build_CAM_model_1D
from dataset import folder_extract, gestures, train_test_split, apply_window
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

    steps = int((len(X_train) / batch_size) * 1.5)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=steps, decay_rate=decay_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, 
                     batch_size=batch_size, callbacks=callbacks, verbose=1)


def main():
    args = parse_args()
    mlflow.tensorflow.autolog()
    mlflow.set_experiment(f"Win{args.window_size} | CAM")

    # Сырые сигналы и метки
    emg, label = folder_extract(FOLDER_PATH, exercises=EXERCISES, myo_pref=MYO_PREF)
    all_g = gestures(emg, label, targets=GESTURE_INDEXES_MAIN)

    # Train/Test split по жестам
    train_g, test_g = train_test_split(all_g, split_size=0.0, rand_seed=GLOBAL_SEED)

    # Преобразование в окна: [N, channels, window]
    X_train_raw, y_train = apply_window(train_g, window=args.window_size, step=STEP_SIZE)
    X_test_raw,  y_test  = apply_window(test_g,  window=args.window_size, step=STEP_SIZE)

    # Каналы для режима и форма входа
    input_shape = (args.window_size, 8)

    # Папка для сохранения параметров стандартизации
    os.makedirs('normalization_values', exist_ok=True)

    # Папка для масок внимания
    os.makedirs('attention_masks', exist_ok=True)

    # Кросс-валидация по фолдам
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    for fold_idx, (idx_tr, idx_vl) in enumerate(kf.split(X_train_raw, y_train), start=1):
        print(f"\n=== Fold {fold_idx} ===")
        Xf_tr, yf_tr = X_train_raw[idx_tr], y_train[idx_tr]
        Xf_vl, yf_vl = X_train_raw[idx_vl], y_train[idx_vl]

        # Считаем mean/std по тренировочным данным (оси 0 и 2)
        means = Xf_tr.mean(axis=(0, 2))       # (channels,)
        stds  = Xf_tr.std(axis=(0, 2)) + 1e-8

        # Сохраняем эти параметры в JSON
        params = {'mean': means.tolist(), 'std':  stds.tolist()}
        norm_file = f'normalization_values/fold{fold_idx}_win{args.window_size}_CAM.json'
        with open(norm_file, 'w') as f:
            json.dump(params, f)

        # Стандартизируем raw-окна
        def standardize(X):
            return (X - means[None,:,None]) / stds[None,:,None]

        Xs_tr = standardize(Xf_tr)
        Xs_vl = standardize(Xf_vl)

        # Переводим в [N, window, channels, 1]
        def prepare(X):
            Xt = np.transpose(X, (0, 2, 1))   # [N, window, channels]
            return Xt.astype(np.float32)

        X_train = prepare(Xs_tr)
        X_valid = prepare(Xs_vl)

        model = build_CAM_model_1D(input_shape, return_attention_mask=True)
        lr = 1e-2    # Для модели с механизмом внимания надо выбирать скорость обучения ниже бейзлайна 

        mflops = get_flops(model, batch_size=1) / 1e6
        print(f"Model MFLOPS: {mflops:.2f}")

        save_w = SAVE_PATH + f'_fold{fold_idx}_{args.window_size}_CAM.h5'
        with mlflow.start_run(run_name=f'fold_{fold_idx}'):
            # Обучение
            train_model(model, EPOCHS, X_train, yf_tr, X_valid, yf_vl, batch_size=BATCH_SIZE, lr=lr, save_path=save_w)
            # Инференс
            model.load_weights(save_w)
            val_acc = model.evaluate(X_valid, yf_vl, batch_size=BATCH_SIZE, verbose=0)[3]
            f1, report_dict, cm_df = evaluate_metrics(model, X_valid, yf_vl)

            mlflow.log_metric('valid_accuracy', float(val_acc))
            mlflow.log_metric('valid_f1', float(f1))
            mlflow.log_metric('complexity_mflops', float(mflops))

            attention_masks = model.predict(X_valid)[1]

            masks_mean = []
            masks_stds = []

            for gest in np.unique(yf_vl):
                attention_masks_gest = attention_masks[np.where(yf_vl == gest)[0]]
                masks_mean.append(attention_masks_gest.mean(axis=(0, 1)))
                masks_stds.append(attention_masks_gest.std(axis=(0, 1)))

            np.save(f'attention_masks/mask_{args.window_size}_fold{fold_idx}.npy', np.array(masks_mean))
            np.save(f'attention_masks/mask_std_{args.window_size}_fold{fold_idx}.npy', np.array(masks_stds))

            # Сохранение матрицы ошибок и отчета
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                cm_df.to_csv(tmp.name)
                mlflow.log_artifact(tmp.name, 'confusion_matrix')
            mlflow.log_dict(report_dict, 'classification_report_valid.json')
            mlflow.log_param('gesture_indexes', GESTURE_INDEXES_MAIN)

        tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
