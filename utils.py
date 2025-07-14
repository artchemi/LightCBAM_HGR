import os
import sys 
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *


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

    probs = model.predict(emg, batch_size=BATCH_SIZE)
    preds = np.argmax(probs, axis=-1)

    f1_score_test = f1_score(labels_true, preds, average='macro', zero_division=0)

    # ! Для сериализации !
    report_test = classification_report(labels_true, preds, output_dict=True, zero_division=0)
    cm_test = confusion_matrix(labels_true, preds)

    labels_unique = np.unique(labels_true)
    cm_df = pd.DataFrame(cm_test, index=labels_unique, columns=labels_unique)

    return f1_score_test, report_test, cm_df
    