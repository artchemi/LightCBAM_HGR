import os
import sys
import pygame
from pygame.locals import *
import multiprocessing
import time
import numpy as np
import json
from collections import deque

from myo_serial import MyoRaw
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from src.models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # ?

# Параметры
WINDOW_SIZE = 56
MODEL_PATH = "emg_recorder/model_finetuned/weights.h5"
STATS_PATH = f"emg_recorder/model_finetuned/fold1_win56_attention.json"
GESTURE_LABELS = GESTURE_INDEXES_MAIN  

# Очереди
data_queue = multiprocessing.Queue()
cmd_queue = multiprocessing.Queue()

# Загрузка модели и статистики
input_shape = (WINDOW_SIZE, len(CHANNELS), 1)
model = build_SAM_model(input_shape, FILTERS_BASE, KERNEL_SIZE_BASE, POOL_SIZE_BASE, P_DROPOUT_BASE, NUM_CLASSES)
model.load_weights(MODEL_PATH)

# model = tf.keras.models.load_model(MODEL_PATH)
with open(STATS_PATH, 'r') as f:
    stats = json.load(f)
mean = np.array(stats["mean"])
std = np.array(stats["std"])

# Myo worker
def worker(data_q, cmd_q):
    m = MyoRaw(raw=True, filtered=False)
    m.connect()

    def add_to_queue(emg, movement):
        data_q.put(emg)

    m.add_emg_handler(add_to_queue)
    m.set_leds([0, 128, 0], [0, 128, 0])
    m.vibrate(1)

    while True:
        while not cmd_q.empty():
            cmd = cmd_q.get()
            if cmd == 'vibrate':
                m.vibrate(1)
            elif cmd == 'stop':
                return
        m.run()

# Отрисовка
def plot(scr, vals):
    global last_vals
    D = 5
    scr.scroll(-D)
    scr.fill((0, 0, 0), (w - D, 0, w, h))

    for i, (u, v) in enumerate(zip(last_vals, vals)):
        pygame.draw.line(scr, (0, 255, 0),
                         (w - D, int(h / 9 * (i + 1 - u))),
                         (w, int(h / 9 * (i + 1 - v))))
        pygame.draw.line(scr, (255, 255, 255),
                         (w - D, int(h / 9 * (i + 1))),
                         (w, int(h / 9 * (i + 1))))
    last_vals = vals

def draw_prediction(scr, prediction):
    pygame.draw.rect(scr, (0, 0, 0), (0, 0, w, 40))
    font = pygame.font.SysFont('Arial', 28)
    text = font.render(f"Prediction: {prediction}", True, (255, 255, 0))
    scr.blit(text, (10, 5))

# Главный цикл
if __name__ == "__main__":
    pygame.init()
    w, h = 800, 600
    scr = pygame.display.set_mode((w, h))
    pygame.display.set_caption("EMG Gesture Predictor")
    clock = pygame.time.Clock()

    # Очередь последних отсчётов
    buffer = deque(maxlen=WINDOW_SIZE)
    global last_vals
    last_vals = [0] * 8

    p_worker = multiprocessing.Process(target=worker, args=(data_queue, cmd_queue))
    p_worker.start()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            # Обновляем буфер
            while not data_queue.empty():
                emg = np.array(data_queue.get())
                buffer.append(emg)
                plot(scr, [e / 20 for e in emg])  # нормализуем для визуализации

            # Если буфер заполнен, делаем предсказание
            if len(buffer) == WINDOW_SIZE:
                window = np.array(buffer)
                standardized = (window - mean) / std
                input_data = np.expand_dims(standardized, axis=0)  # (1, 56, 8)
                pred = model.predict(input_data, verbose=0)
                gesture = GESTURE_LABELS[np.argmax(pred)]

                draw_prediction(scr, gesture)

            pygame.display.flip()
            clock.tick(200)

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        cmd_queue.put('stop')
        p_worker.join()
        pygame.quit()
        sys.exit()
