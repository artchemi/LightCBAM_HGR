import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from collections import defaultdict

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

# Пример использования
if __name__ == "__main__":
    col_name_accuracy = "validation_softmax_accuracy"
    col_name_f1 = "valid_f1"
    col_name_complexity = "complexity_mflops"
    model = 'CAM'

    window_sizes = np.arange(24, 101, 4)
    metrics_dict = defaultdict(list)
    metrics_dict['window_size'] = window_sizes

    for i in window_sizes:
        df_accuracy, _ = average_metric_for_experiment(f"Win{i}|{model}|reduced", col_name_accuracy)
        df_f1, _ = average_metric_for_experiment(f"Win{i}|{model}|reduced", col_name_f1)
        df_complexity, _ = average_metric_for_experiment(f"Win{i}|{model}|reduced", col_name_complexity)

        metrics_dict['accuracy_mean'].append(df_accuracy[col_name_accuracy].mean() * 100)
        metrics_dict['accuracy_std'].append(df_accuracy[col_name_accuracy].std() * 100)

        metrics_dict['f1_mean'].append(df_f1[col_name_f1].mean())
        metrics_dict['f1_std'].append(df_f1[col_name_f1].std())

        metrics_dict[col_name_complexity].append(df_complexity[col_name_complexity].iloc[0])

    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv(f'metrics_{model}.csv', index=False)
