import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import scipy.io
import random
import os, sys
from collections import defaultdict
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import GESTURE_INDEXES_MAIN, COM_PORTS


def folder_extract(root_dir, exercises=["E2"]):
    """
    Purpose:
        Extract sEMG signals data from files beneath folder 'root_dir'(from args)

    Args:
        1. root_dir (str):
            Root directory of the Ninapro DB5. (With folders and files storing sEMG data underneath)
        
        2. exercises (1D list, optional):
            Exercises with dedicated gestures stored. Defaults to "E2".
            - Note:
                "E3" may match to file: Ninapro_DB5\s2\S2_E3_A1.mat
                "E2" may match to file: Ninapro_DB5\s2\S2_E2_A1.mat
                "E1" may match to file: Ninapro_DB5\s2\S2_E1_A1.mat
            - Example:
                ["E3", "E2"] as args collects sample from both exercise 3 and 2.
        
        3. myo_pref (str, optional):
            Ninapro DB5 data was collected via 2 Myo armband
            - "elbow" collects sEMG from 1:8 channels, samples closest to elbow (From Myo Armband 1)
            - "wrist" collects sEMG from 9:16 channels, samples closest to wrist (From Myo Armband 2)

            Defaults to "elbow".

    Returns:
        1. (numpy.ndarray):
            - Samples collected from "emg" column within each .mat files wihtin the folder 'root_dir'(from args)
            - Shape: [num samples, 8(1 sEMG sample from each 8 Myo sensors/channels)]
            
        2. (numpy.ndarray): _description_
            - Targets/labels collected from "stimulus" column within each .mat files wihtin the folder 'root_dir'(from args)
            - Shape: [num samples]
    """
    
    emg = []
    emg_label = []
    
    # Parse through sub folders underneath 'root_dir'(from args)
    for folder in tqdm(os.listdir(root_dir)):
        subfolder_dir = root_dir + "/" + folder
        # Parse through .mat files underneath sub folders
        for file in os.listdir(subfolder_dir):
            # Get sEMG signals of dedicated Myo armband and Exercise
            if file.split("_")[1] in exercises:
                file_path = subfolder_dir + "/" + file
                # Read .mat file
                mat = scipy.io.loadmat(file_path)
                
                emg += [sensors[:8] for sensors in mat["emg"]]

                current_exercise = file.split("_")[1]
                
                if current_exercise == "E2":
                    labels = mat["stimulus"].reshape(-1)
                    new_labels = []
                    
                    for label in labels:
                        if label != 0:
                            new_labels.append(label + 12)
                        else:
                            new_labels.append(0)
                    
                    emg_label.extend(new_labels)
                
                elif current_exercise == "E3":
                    labels = mat["stimulus"].reshape(-1)
                    new_labels = []
                    
                    for label in labels:
                        if label != 0:
                            new_labels.append(label + 29)
                        else:
                            new_labels.append(0)
                    
                    emg_label.extend(new_labels)
                
                else:
                    # Collect corresponding labels
                    emg_label.extend(mat["stimulus"].reshape(-1))
    
    return np.array(emg), np.array(emg_label)

def folder_extract_subject(root_dir: str, exercises: list=["E2"], subjects: list=['s1']) -> list:
    """Извлекает серии ЭМГ сигналов для всех субъектов в списке subjects.

    Args:
        root_dir (str): Корневая папка с датасетом.
        exercises (list, optional): Номера упражнений (см. оригинальный датасет). Defaults to ["E2"].
        subjects (list, optional): Номера активных субъектов. Defaults to ['s1'].

    Returns:
        list: Список массивов ЭМГ сигналов и меток.
    """
    emg = []
    emg_label = []
    files_lst = os.listdir(root_dir)                                    # Список файлов
    files_lst = [file for file in files_lst if file in subjects]    # Выделение только необходимых пользователей
    files_lst = sorted(files_lst, key=lambda x: int(x[1:]))             # Сортировка названий
    
    for folder in tqdm(files_lst, desc=f'Извлечение сигналов для {subjects}'):
        subfolder_dir = root_dir + "/" + folder    # Определение субдиректории с пользователем
        for file in os.listdir(subfolder_dir):     # Перебор всех пользователей
            if file.split("_")[1] in exercises:
                file_path = subfolder_dir + "/" + file

                mat = scipy.io.loadmat(file_path)                 # Чтение .mat
                emg += [sensors[:8] for sensors in mat["emg"]]    # ! Извлечение сырых сигналов

                current_exercise = file.split("_")[1]
                
                if current_exercise == "E2":
                    labels = mat["stimulus"].reshape(-1)
                    new_labels = []
                    
                    for label in labels:
                        if label != 0:
                            new_labels.append(label + 12)    # Увеличение индекса для соответсвия таблице
                        else:
                            new_labels.append(0)
                    
                    emg_label.extend(new_labels)
                
                elif current_exercise == "E3":
                    labels = mat["stimulus"].reshape(-1)
                    new_labels = []
                    
                    for label in labels:
                        if label != 0:
                            new_labels.append(label + 29)    # Увеличение индекса для соответсвия таблице
                        else:
                            new_labels.append(0)
                    emg_label.extend(new_labels)

                else:
                    emg_label.extend(mat["stimulus"].reshape(-1))
    
    return np.array(emg), np.array(emg_label)

def recordings_extract(root_dir: str="../dataset_unified", trials: list=["1", "2", "3"], 
                       gestures: list=GESTURE_INDEXES_MAIN, COM_ports: list=COM_PORTS, subjects: list=["S1"]) -> tuple:
    """Извлечение данных из .csv файлов с директории dataset_unified
    
    Args:
        root_dir (str, опционально): Корневая папка датасета. По умолчанию: "../dataset_unified".
        trials (list, опционально): Индекс запуска. 1 - слабые усилия, 2 - средние, 3 - сильные. По умолчанию: ["1", "2", "3"].
        gestures (list, опционально): Индексы жестов. См. документацию. По умолчанию: [0, 13, 15, 18, 19, 34, 38, 43, 46]
        subjects (list, опционально): Индекс субъектов. По умолчанию ["S1"].

    Returns:
        tuple: Кортеж массивов np.ndarray: 
        1-й массив - ЭМГ сигналы размерностью (n, c), где n - суммарное кол-во точек, c - количество каналов;
        2-й массив - метки классов, размерностью (n, ).
    """
    files = [s for s in sorted(os.listdir(root_dir)) if (s.startswith("S")) and ("table" not in s)]    # Все файлы .csv с сигналами
    filtered_columns = ["EMG_FILTERED_" + com for com in COM_ports]                                    # Названия колонок с отфильтрованными ЭМГ
    
    emgs = []
    labels = []

    files_missed = []    # Файлы, в которых пропущены колонки из-за ошибки

    for gest in tqdm(GESTURE_INDEXES_MAIN):
        for s in subjects:
            for trial in trials:
                file_result = [file for file in files if f"{s}_{gest}_{trial}" in file][0]             # Искомое название файла
                df_main = pd.read_csv(root_dir + file_result, comment="#").dropna()                    # .csv с сериями ЭМГ                    

                try:                                                      # Обработчик исключений на наличие колонок filtered_columns в датасете
                    df_filtered = df_main[filtered_columns].iloc[500:]    # Обрезанные и отфильтрованные сигналы
                except:
                    files_missed.append(file_result)
                    continue

                emgs.extend(df_filtered.to_numpy())
                labels.extend([gest]*df_filtered.shape[0])
    
    print(f"Skippped dataframes: {files_missed}")

    return np.asarray(emgs), np.asarray(labels)


def data2gestdict(emgs: np.ndarray, labels: np.ndarray) -> Dict[int, list]:    # NOTE: Для несбалансированных классов можно добавить thrs/relax_shrink
    """Переводит массивы ЭМГ сигналов и меток классов жестов в словарь.

    Args:
        emgs (np.ndarray): N-мерный массив ЭМГ сигналов, где N - количество каналов.
        labels (np.ndarray): Одномерный массив меток.

    Returns:
        dict: `{0: [...], 1: [...], 2: [...], ...}`, где ключи - индекс жестов, значения - списки массивов np.ndarray.
    """
    unique_labels = np.unique(labels)    # Уникальные метки жестов
    gesture_dict = defaultdict(list)

    label_indexes = np.where(labels == unique_labels[1])

    for label_i in unique_labels:
        label_indexes = np.where(labels == label_i)          # Индексы меток i в labels
        gesture_dict[label_i].extend(emgs[label_indexes])    # Добавление в список ЭМГ сигналов с индексами label_indexes
        
    return gesture_dict


def train_test_split(gestures: dict, split_size: float=0.25, rand_seed: int=42) -> List[Dict[int, list]]:
    """Разделяет данные на тренировочные и тестовые.

    Args:
        gestures (dict): Общий словарь жестов.
        split_size (float, optional): Размер тестовой выборки. По умолчанию: 0.25.
        rand_seed (int, optional): Случайный seed. По умолчанию: 42.

    Returns:
        List[Dict[int, list]]: Список двух словарей с тренировочной и тестовой выборкой соответственно. 
    """
    train_gestures = {key:None for key in gestures}
    test_gestures = {key:None for key in gestures}
    
    for _, (label, signals) in enumerate(gestures.items()):
        random.Random(rand_seed).shuffle(signals)    # Перемешать индексы
        
        threshold = int(len(signals) * split_size)
        
        train_gestures[label] = signals[threshold:]
        test_gestures[label] = signals[:threshold]
    
    return train_gestures, test_gestures


def standarization(emg, save_path=None):
    """
    Purpose:
        Apply Standarization (type feature scaling) to sEMG samples 'emg'(from args)

    Args:
        1. emg (numpy.ndarray):
            The sEMG samples to apply Standarization (First output of function "folder_extract")
            
        2. save_path (str, optional):
            Path of json storing MEAN and Standard Deviation for each sensor Channel. Defaults to None.

    Returns:
        (numpy.ndarray):
            sEMG signals scaled with Standarization.
    """

    # Dictionary storing MEAN and Standard Deviation for each sensor Channel
    params = {i:[None, None] for i in range(8)}
    
    # Transform shape of 'emg'(from args)
    # [num samples, 8(sensors/channels)] -> [8(sensors/channels), num samples]
    new_emg = []
    for channel_idx in range(8):
        # Collect all samples of each sensor/channel
        new_emg.append([emg_arr[channel_idx] for _, emg_arr in enumerate(emg)])
    new_emg = np.array(new_emg)
    
    # Apply Standarization
    for channel_idx in range(8):
        # Calculate Mean from samples of each local sensor/channel
        params[channel_idx][0] = float(np.mean(new_emg[channel_idx]))
        # Calculate Standard Deviation from samples of each local sensor/channel
        params[channel_idx][1] = float(np.std(new_emg[channel_idx]))
        # Apply Standarization to samples of each local sensor/channel
        new_emg[channel_idx] = (new_emg[channel_idx] - params[channel_idx][0])/params[channel_idx][1]
    
    # Transform shape of new_emg
    # [8(sensors/channels), num samples] -> [num samples, 8(sensors/channels)]
    final_emg = []
    for idx in range(new_emg.shape[1]):
        # Convert back to sEMG arrays with 1 sample from each sensor/channel
        final_emg.append([sensor_samples[idx] for _, sensor_samples in enumerate(new_emg)])
    final_emg = np.array(final_emg)
    
    # Save MEANs and Standard Deviations if 'save_path'(from args) was provided
    if save_path != None:
        with open(save_path, 'w') as f:
            json.dump(params, f)
    
    return np.array(final_emg)


def gestures(emg, label, targets=[0, 1, 3, 6],
             relax_shrink=80000, rand_seed=2022):
    """
    Purpose:
        Organize sEMG samples to dictionary with:
            - key: gesture/label
            - values: array of sEMG sigals corresponding to the specific gesture/label

    Args:
        1. emg (numpy.ndarray):
            The array of sEMG samples (First output of function "folder_extract" or "standarization")
        
        2. label (numpy.ndarray):
            Array of labels for the sEMG samples (Second output of function "folder_extract")
        
        3. targets (list, optional):
            Array of specified wanted gesture/label. Defaults to [0, 1, 3, 6].
        
        4. relax_shrink (int, optional): Shrink size for relaxation gesture. Defaults to 80000.
        
        5. rand_seed (int, optional): Random seed for shuffling before shrinking relaxation gesture samples. Defaults to 2022.

    Returns:
        gestures (dict):
            - Dictionary with:
                - key: gesture/label
                - values: array of sEMG sigals corresponding to the gesture/label
                
            - Structure:
                {
                    0 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    1 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    ...
                    num gestures (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                }
    """
    
    if relax_shrink != None:
        assert 0 in targets
        assert rand_seed != None
    
    gestures = {label:[] for label in targets}
    # Sort each sEMG array to the corresponding gesture/label
    for idx, emg_array in enumerate(emg):
        if label[idx] in gestures:
            gestures[label[idx]].append(emg_array)
    
    # Too much relaxation gesture, just randomly shrink some
    if relax_shrink != None:
        random.seed(rand_seed)
        gestures[0] = random.sample(gestures[0], relax_shrink)
    
    return gestures
    
    
def apply_window(gestures, window=32, step=16):
    """
    Purpose:
        Convert sEMG signal samples to sEMG image format.

    Args:
        1. gestures (dict):
            (Any output from function "gestures" or "train_test_split")
        
            - Dictionary with:
                - key: gesture/label
                - values: array of sEMG sigals corresponding to the gesture/label
            - Structure:
                {
                    0 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    1 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    ...
                    num gestures (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                }
                
        2. window (int, optional):
            How many samples each sEMG image channel contains. Defaults to 52.

    Returns:
        1. signals (numpy.ndarray):
            Processed sEMG signals in sEMG image format.
            - Example shape: [num samples, 1, 8(sensors/channels), 52(window)]
            
        2. outputs (numpy.ndarray):
            Labels for the sEMG signals
    """
    inputs = []
    outputs = []

    # Segment samples to list of windows
    for idx, (label, signals) in enumerate(gestures.items()):
        # signals.shape: [num samples, 8(sensors/channels)]
        signals = np.array(signals)
            
        windowed_signals = [signals[i:i+window] for i in range(0, len(signals)-window, step)]
        
        inputs.extend(windowed_signals)
        outputs.extend(
            [idx for _ in range(len(windowed_signals))]    
        )

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    
    signals = []

    # Transform dimensions:
    #   [num samples, window, sensors/channels] -> [num samples, sensors/channels, window]
    for samples in inputs:
        # sample.shape: [window, sensors/channels]
        
        temp_window = []
        for channel_idx in range(len(samples[0])):
            # Collect channel/sensor sample from each emg_array
            temp_window.append([emg_array[channel_idx] for _, emg_array in enumerate(samples)])
            
        signals.append(temp_window)

    signals = np.array(signals)
    
    return signals, outputs


def realtime_preprocessing(emg, params_path=None, num_classes=4, window=32, step=16):
    """
    Purpose:
        Preprocess data samples obtained from realtime.py

    Args:
        1. emg (list):
            The sEMG samples obtained from realtime.py
        
        2. params_path (list, optional):
            - Path of json storing MEAN and Standard Deviation for each sensor Channel. Defaults to None.
        
        3. num_classes (int, optional):
            - Number of gestures/classes the new finetune model would like to classify. Defaults to 4.

    Returns:
        1. inputs (numpy.ndarray):
            Processed sEMG signals in sEMG image format.
            - Example shape: [num samples, 1, 8(sensors/channels), 52(window)]
        2. outputs (numpy.ndarray):
            Labels for the sEMG signals
    """
    emg = np.array(emg)
    
    # Apply Standarization feature scaling to samples if 'params_path'(from args) was provided
    if params_path != None:
        scaled_signals = []
        
        with open(params_path, 'r') as f:
            params = json.load(f)
            
        for channel_idx in range(8):
            mean = params[str(channel_idx)][0]
            std = params[str(channel_idx)][1]
            
            current_sample = emg[channel_idx]
            
            scaled_signals.append(
                (current_sample - mean) / std
            )
        scaled_signals = np.array(scaled_signals)
    else:
        scaled_signals = np.array(emg)
    
    # Convert sEMG sampels to sEMG windows appropriate for training
    sEMG = []
    for i in range(len(scaled_signals[0])):
        sEMG.append([scaled_signals[channel_idx][i] for channel_idx in range(8)])
    
    gesture = {i:[] for i in range(num_classes)}
    curr_gest = 0
    gest_size = int(len(sEMG)/num_classes)
    
    for i in range(0, len(sEMG), gest_size):
        gesture[curr_gest] = sEMG[i:i+gest_size]
        curr_gest += 1
    
    inputs, outputs = apply_window(gesture, window, step)
    
    return inputs, outputs