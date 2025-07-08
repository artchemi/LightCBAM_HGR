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
from models import build_base_model, build_SAM_model
from dataset import folder_extract, gestures, apply_window, standarization

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from utils import *

set_seed(seed=GLOBAL_SEED)

print(tf.config.list_physical_devices('GPU'))
tf.get_logger().setLevel('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_args():
    parser = argparse.ArgumentParser(description='Обучение модели классификации жестов')
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE, help='Размер окна')
    parser.add_argument('--mode', type=str, default='base', help='Режим запуска эксперимента: base, reduced, attention')
    return parser.parse_args()

def train_model(model, epochs, X_train, y_train, X_valid, y_valid, batch_size=BATCH_SIZE, lr=INIT_LR,
                decay_rate=0.9, save_path=SAVE_PATH, patience=PATIENCE):
    callbacks = []
    if save_path:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            save_path, monitor='val_loss', verbose=1, save_best_only=True,
            mode='min', save_weights_only=True)
        callbacks.append(checkpoint)
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience))

    steps = (len(X_train) / batch_size) * 1.5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=steps, decay_rate=decay_rate)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy', metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train, validation_data=(X_valid, y_valid),
        batch_size=batch_size, epochs=epochs, callbacks=callbacks
    )
    return history

def main():
    args = parse_args()
    mlflow.tensorflow.autolog()
    mlflow.set_experiment(f"Window Size {args.window_size} | {args.mode}")

    emg, label = folder_extract(FOLDER_PATH, exercises=EXERCISES, myo_pref=MYO_PREF)
    all_gestures = gestures(emg, label, targets=GESTURE_INDEXES_MAIN)
    print(f'Выбранные жесты: {list(all_gestures.keys())}')

    all_data, all_labels = apply_window(all_gestures, window=args.window_size, step=STEP_SIZE, return_labels=True)
    all_data, all_labels = np.array(all_data), np.array(all_labels)

    kf = StratifiedKFold(n_splits=5)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(all_data, all_labels), 1):
        print(f'=== Fold {fold} ===')

        X_train_raw, y_train = all_data[tr_idx], all_labels[tr_idx]
        X_val_raw, y_val = all_data[val_idx], all_labels[val_idx]

        # Стандартизация по train only
        means = np.mean(X_train_raw, axis=(0, 1))
        stds = np.std(X_train_raw, axis=(0, 1)) + 1e-8
        X_train = (X_train_raw - means) / stds
        X_val = (X_val_raw - means) / stds

        # Выбор каналов и reshape
        channels = [0, 3, 4, 5, 6] if args.mode == 'reduced' else list(range(8))
        def reshape(X):
            X = X.reshape(-1, 8, args.window_size, 1)
            X = np.transpose(X, (0, 2, 1, 3))
            return X[:, :, channels, :].astype(np.float32)

        X_train, X_val = reshape(X_train), reshape(X_val)
        input_shape = (args.window_size, len(channels), 1)

        if args.mode in ['base', 'reduced']:
            model = build_base_model(input_shape, FILTERS_BASE, KERNEL_SIZE_BASE, POOL_SIZE_BASE, P_DROPOUT_BASE, NUM_CLASSES)
            lr = INIT_LR
        else:
            model = build_SAM_model(input_shape, FILTERS_BASE, KERNEL_SIZE_BASE, POOL_SIZE_BASE, P_DROPOUT_BASE, NUM_CLASSES)
            lr = 1e-2
        mflops = get_flops(model, batch_size=1) / 1e6

        with mlflow.start_run(run_name=f'fold_{fold}'):
            train_model(model, EPOCHS, X_train, y_train, X_val, y_val, save_path=SAVE_PATH + f'_{args.window_size}_{args.mode}_fold{fold}.h5', lr=lr)

            model.load_weights(filepath=SAVE_PATH + f'_{args.window_size}_{args.mode}_fold{fold}.h5')
            _, valid_acc = model.evaluate(X_val, y_val, verbose=0)
            f1_valid, report_valid_dict, cm_valid_df = evaluate_metrics(model, X_val, y_val)

            mlflow.log_metric('valid_accuracy', float(valid_acc))
            mlflow.log_metric('complexity', mflops)
            mlflow.log_metric("valid_f1", f1_valid)

            with tempfile.NamedTemporaryFile(suffix=".csv", mode='w', delete=False) as f:
                cm_valid_df.to_csv(f.name)
                mlflow.log_artifact(f.name, artifact_path="confusion_matrix")

            mlflow.log_param("gesture_indexes", GESTURE_INDEXES_MAIN)
            mlflow.log_dict(report_valid_dict, "classification_report_valid.json")

        tf.keras.backend.clear_session()

if __name__ == '__main__':
    main()
