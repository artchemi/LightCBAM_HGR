import os
import sys
import pygame
from pygame.locals import *
import multiprocessing
import time
import numpy as np

from myo_serial import MyoRaw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_recorder import *

os.makedirs('emg_recorder/data/discrete', exist_ok=True)
# Очереди 
data_queue = multiprocessing.Queue()
cmd_queue = multiprocessing.Queue()

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Сбор ЭМГ сигналов')
    parser.add_argument('--gesture', type=int, default=0, help='Номер жеста')
    return parser.parse_args()

# Myo worker 
def worker(data_q, cmd_q):
    m = MyoRaw(raw=True, filtered=False)
    m.connect()

    def add_to_queue(emg, movement):
        data_q.put(emg)

    m.add_emg_handler(add_to_queue)

    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    m.set_leds([128, 0, 0], [128, 0, 0])
    m.vibrate(1)

    while True:
        # Проверяем команды
        while not cmd_q.empty():
            cmd = cmd_q.get()
            if cmd == 'vibrate':
                m.vibrate(1)
            elif cmd == 'stop':
                return  # завершаем воркер

        m.run()

last_vals = None

# Отрисовка EMG-графика 
def plot(scr, vals):
    global last_vals
    DRAW_LINES = True

    if last_vals is None:
        last_vals = vals
        return

    D = 5
    scr.scroll(-D)
    scr.fill((0, 0, 0), (w - D, 0, w, h))

    for i, (u, v) in enumerate(zip(last_vals, vals)):
        if DRAW_LINES:
            pygame.draw.line(scr, (0, 255, 0),
                             (w - D, int(h / 9 * (i + 1 - u))),
                             (w, int(h / 9 * (i + 1 - v))))
            pygame.draw.line(scr, (255, 255, 255),
                             (w - D, int(h / 9 * (i + 1))),
                             (w, int(h / 9 * (i + 1))))
    last_vals = vals

# Отрисовка таймера 
def draw_timer(scr, start_time):
    pygame.draw.rect(scr, (0, 0, 0), (0, 0, 150, 30))
    font = pygame.font.SysFont('Arial', 24)
    elapsed_time = time.time() - start_time
    text = font.render(f"Time: {elapsed_time:.1f}s", True, (255, 255, 0))
    scr.blit(text, (10, 5))

#  Стартовый экран 
def draw_start_screen(scr):
    scr.fill((30, 30, 30))
    font = pygame.font.SysFont('Consolas', 36)
    text = font.render("Press to record EMG", True, (255, 255, 255))
    scr.blit(text, (w // 2 - text.get_width() // 2, h // 3))

    # Кнопка
    button_rect = pygame.Rect(w // 2 - 100, h // 2, 200, 60)
    pygame.draw.rect(scr, (70, 130, 180), button_rect)
    pygame.draw.rect(scr, (255, 255, 255), button_rect, 2)

    button_text = font.render("Start", True, (255, 255, 255))
    scr.blit(button_text, (
        button_rect.centerx - button_text.get_width() // 2,
        button_rect.centery - button_text.get_height() // 2
    ))

    pygame.display.flip()
    return button_rect


if __name__ == "__main__":
    args = parse_args()
    pygame.init()

    p_worker = multiprocessing.Process(target=worker, args=(data_queue, cmd_queue))
    p_worker.start()

    w, h = W, H
    scr = pygame.display.set_mode((w, h))
    pygame.display.set_caption("EMG Recorder")

    start_time = None
    recording = False
    button_rect = draw_start_screen(scr)

    clock = pygame.time.Clock()

    recorded_data = []  # Контейнер для сбора ЭМГ

    # ! Основной цикл !
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

                if event.type == pygame.MOUSEBUTTONDOWN and not recording:
                    if button_rect.collidepoint(event.pos):
                        start_time = time.time()
                        recording = True
                        scr.fill((0, 0, 0))
                        cmd_queue.put('vibrate')

            if recording:
                elapsed_time = time.time() - start_time

                if elapsed_time > DURATION:
                    scr.fill((255, 255, 255))
                    pygame.display.flip()

                    # Сохраняем данные в .npy
                    if recorded_data:
                        np.save(f"emg_recorder/data/discrete/emg_data_{args.gesture}.npy", np.array(recorded_data))
                        print(f"Saved {len(recorded_data)} EMG samples to emg_data.npy")

                    recording = False
                    # Отправляем команду на остановку воркера
                    cmd_queue.put('stop')
                    break

                emg = None

				# Сбор всех данных в пакете
                while not data_queue.empty():
                    emg = data_queue.get()
                    recorded_data.append(emg)
                    plot(scr, [e / PLOT_REDUCTION for e in emg])
                    
                draw_timer(scr, start_time)
                pygame.display.flip()

            clock.tick(200)

    except KeyboardInterrupt:
        print("Quitting")

    finally:
        if recording and recorded_data:
            np.save("emg_data.npy", np.array(recorded_data))
            print(f"Saved {len(recorded_data)} EMG samples to emg_data.npy")

        cmd_queue.put('stop')
        p_worker.join()

        pygame.quit()
        sys.exit()
