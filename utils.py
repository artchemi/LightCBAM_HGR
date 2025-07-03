import os
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def set_seed(seed=42):
    """Фиксация сидов для воспроизводимости.

    Args:
        seed (int, optional): _description_. Defaults to 42.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass


def evaluate_metrics(model: tf.keras.Sequential, emg: np.ndarray, labels_true: np.ndarray):

    labels_preds = model.predict(emg)

    f1_score_test = f1_score(labels_true, labels_preds, average='macro', zero_division=0)
    classification_report_test = classification_report(labels_true, labels_preds, zero_division=0)

    return f1_score_test, classification_report_test
    