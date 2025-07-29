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
    parser = argparse.ArgumentParser(description='Сбор EMG+IMU сигналов')
    parser.add_argument('--gesture', type=int, default=0, help='Номер жеста')
    return parser.parse_args()

# Myo worker
def worker(data_q, cmd_q):
    m = MyoRaw(raw=True, filtered=False)
    m.connect()

    # EMG → очередь
    def add_emg_to_queue(emg, movement):
        data_q.put(('emg', np.array(emg)))
    m.add_emg_handler(add_emg_to_queue)

    # IMU → очередь
    def add_imu_to_queue(quat, acc, gyro):
        data_q.put(('imu', np.array(acc)))  # сохраняем акселерометр
    m.add_imu_handler(add_imu_to_queue)

    def print_battery(bat):
        print("Battery level:", bat)
    m.add_battery_handler(print_battery)

    m.set_leds([128, 0, 0], [128, 0, 0])
    m.vibrate(1)

    while True:
        while not cmd_q.empty():
            cmd = cmd_q.get()
            if cmd == 'vibrate':
                m.vibrate(1)
            elif cmd == 'stop':
                return
        m.run()

last_vals = None

def plot(scr, vals):
    global last_vals
    if last_vals is None:
        last_vals = vals
        return

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


def draw_timer(scr, start_time):
    pygame.draw.rect(scr, (0, 0, 0), (0, 0, 150, 30))
    font = pygame.font.SysFont('Arial', 24)
    elapsed = time.time() - start_time
    text = font.render(f"Time: {elapsed:.1f}s", True, (255, 255, 0))
    scr.blit(text, (10, 5))


def draw_start_screen(scr):
    scr.fill((30, 30, 30))
    font = pygame.font.SysFont('Consolas', 36)
    text = font.render("Press to record", True, (255, 255, 255))
    scr.blit(text, (w//2 - text.get_width()//2, h//3))

    button = pygame.Rect(w//2 -100, h//2, 200, 60)
    pygame.draw.rect(scr, (70, 130, 180), button)
    pygame.draw.rect(scr, (255, 255, 255), button, 2)
    btn_text = font.render("Start", True, (255, 255, 255))
    scr.blit(btn_text, (button.centerx - btn_text.get_width()//2,
                        button.centery - btn_text.get_height()//2))
    pygame.display.flip()
    return button

if __name__ == "__main__":
    args = parse_args()
    pygame.init()

    p = multiprocessing.Process(target=worker, args=(data_queue, cmd_queue))
    p.start()

    w, h = W, H
    scr = pygame.display.set_mode((w, h))
    pygame.display.set_caption("EMG+IMU Recorder")

    start_time = None
    recording = False
    button = draw_start_screen(scr)
    clock = pygame.time.Clock()

    recorded = []  # список конкатенированных сэмплов
    last_imu = np.zeros(3)

    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if ev.type == pygame.MOUSEBUTTONDOWN and not recording:
                    if button.collidepoint(ev.pos):
                        start_time = time.time()
                        recording = True
                        scr.fill((0,0,0))
                        cmd_queue.put('vibrate')

            if recording:
                elapsed = time.time() - start_time
                if elapsed > DURATION:
                    scr.fill((255,255,255)); pygame.display.flip()
                    arr = np.vstack(recorded)
                    np.save(f"emg_recorder/data/discrete/emg_imu_{args.gesture}.npy", arr)
                    print(f"Saved array shape {arr.shape}")
                    recording=False
                    cmd_queue.put('stop')
                    break

                # обрабатываем очередь
                while not data_queue.empty():
                    typ, payload = data_queue.get()
                    if typ == 'imu':
                        last_imu = payload
                    else:  # 'emg'
                        emg = payload
                        combined = np.hstack((emg, last_imu))
                        recorded.append(combined)
                        plot(scr, emg / PLOT_REDUCTION)

                draw_timer(scr, start_time)
                pygame.display.flip()
            clock.tick(200)

    except KeyboardInterrupt:
        pass
    finally:
        if recording and recorded:
            arr = np.vstack(recorded)
            np.save("emg_imu_backup.npy", arr)
            print(f"Saved backup shape {arr.shape}")
        cmd_queue.put('stop')
        p.join()
        pygame.quit()
        sys.exit()
