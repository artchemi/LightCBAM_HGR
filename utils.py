import os
import sys 
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import mlflow
from mlflow.tracking import MlflowClient
from collections import defaultdict

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

    if (type(probs) == tuple) | (type(probs) == list):
        probs = probs[0]

    preds = np.argmax(probs, axis=-1)

    f1_score_test = f1_score(labels_true, preds, average='macro', zero_division=0)

    # ! Для сериализации !
    report_test = classification_report(labels_true, preds, output_dict=True, zero_division=0)
    cm_test = confusion_matrix(labels_true, preds)

    labels_unique = np.unique(labels_true)
    cm_df = pd.DataFrame(cm_test, index=labels_unique, columns=labels_unique)

    return f1_score_test, report_test, cm_df

def average_metric_for_experiment(experiment_name: str, metric_key: str):
    client = MlflowClient()
    # Получаем информацию об эксперименте
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Эксперимент с именем '{experiment_name}' не найден.")
    exp_id = experiment.experiment_id

    # Достаем все запуски эксперимента
    runs = client.search_runs(experiment_ids=[exp_id],
                              filter_string="",
                              run_view_type=mlflow.entities.ViewType.ALL)
    if not runs:
        raise ValueError(f"В эксперименте '{experiment_name}' нет запусков.")

    # Собираем значения метрики metric_key из каждого запуска
    data = []
    for run in runs:
        run_id = run.info.run_id
        # Берём последнее зарегистрированное значение метрики
        metric = client.get_metric_history(run_id, metric_key)
        if not metric:
            continue
        last_value = metric[-1].value
        data.append({
            "run_id": run_id,
            metric_key: last_value
        })

    # Формируем DataFrame и считаем среднее
    df = pd.DataFrame(data)
    mean_val = df[metric_key].mean()
    # print(f"Усредненное значение метрики '{metric_key}' по эксперименту '{experiment_name}': {mean_val:.4f}")
    return df, mean_val
    