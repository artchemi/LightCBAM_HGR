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

os.makedirs('emg_recorder/data/continuous/', exist_ok=True)
# Очереди 
data_queue = multiprocessing.Queue()
cmd_queue = multiprocessing.Queue()

TEXT_AREA_HEIGHT = 60
GRAPH_WIDTH = 800
GRAPH_HEIGHT = 540  # h - TEXT_AREA_HEIGHT
GRAPH_SURFACE = pygame.Surface((GRAPH_WIDTH, GRAPH_HEIGHT))
GRAPH_SURFACE.fill((0, 0, 0))

graph_x = 0

gesture_dict_labels = {'#13': 1, '#15': 2, '#18': 3, '#19': 4, 
                       '#34': 5, '#38': 6, '#43': 7, '#46': 8}

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
        while not cmd_q.empty():
            cmd = cmd_q.get()
            if cmd == 'vibrate':
                m.vibrate(1)
            elif cmd == 'stop':
                return
        m.run()

# Глобальные переменные 
last_vals = None

def plot(scr, vals):
    global last_vals, GRAPH_SURFACE, graph_x

    if last_vals is None:
        last_vals = vals
        return

    GRAPH_SURFACE.scroll(-1, 0)
    pygame.draw.rect(GRAPH_SURFACE, (0, 0, 0), (GRAPH_WIDTH - 1, 0, 1, GRAPH_HEIGHT))

    for i, (u, v) in enumerate(zip(last_vals, vals)):
        y1 = int(GRAPH_HEIGHT / 9 * (i + 1 - u))
        y2 = int(GRAPH_HEIGHT / 9 * (i + 1 - v))

        pygame.draw.line(GRAPH_SURFACE, (0, 255, 0), (GRAPH_WIDTH - 2, y1), (GRAPH_WIDTH - 1, y2))
        pygame.draw.line(GRAPH_SURFACE, (255, 255, 255),
                         (GRAPH_WIDTH - 2, int(GRAPH_HEIGHT / 9 * (i + 1))),
                         (GRAPH_WIDTH - 1, int(GRAPH_HEIGHT / 9 * (i + 1))))

    last_vals = vals
    scr.blit(GRAPH_SURFACE, (0, TEXT_AREA_HEIGHT))

def draw_start_screen(scr):
    scr.fill((30, 30, 30))
    font = pygame.font.SysFont('Consolas', 36)
    text = font.render("Press to record EMG", True, (255, 255, 255))
    scr.blit(text, (w // 2 - text.get_width() // 2, h // 3))

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
    pygame.init()

    p_worker = multiprocessing.Process(target=worker, args=(data_queue, cmd_queue))
    p_worker.start()

    w, h = W, H
    scr = pygame.display.set_mode((w, h))
    pygame.display.set_caption("EMG Recorder")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 36)

    button_rect = draw_start_screen(scr)
    recording = False

    # ✅ Разделяем сигналы и метки
    recorded_emg = []
    gesture_labels = []

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.MOUSEBUTTONDOWN and not recording:
                    if button_rect.collidepoint(event.pos):
                        cmd_queue.put('vibrate')
                        recording = True
                        gesture_idx = 0
                        gesture_start = time.time()
                        scr.fill((0, 0, 0))

            if recording:
                if gesture_idx >= len(GESTURES):
                    # Сохраняем отдельно данные и метки
                    np.save("emg_recorder/data/continuous/emg_data_full.npy", np.array(recorded_emg))
                    np.save("emg_recorder/data/continuous/gesture_labels_full.npy", np.array(gesture_labels))
                    print(f"Saved {len(recorded_emg)} samples and labels")
                    cmd_queue.put('stop')
                    break

                gesture_name, gesture_duration = GESTURES[gesture_idx]
                elapsed = time.time() - gesture_start

                if elapsed > gesture_duration:
                    gesture_idx += 1
                    gesture_start = time.time()
                    scr.fill((0, 0, 0))
                    continue

                pygame.draw.rect(scr, (0, 0, 0), (0, 0, w, TEXT_AREA_HEIGHT))
                text = font.render(f"Gesture: {gesture_name} | Time left: {gesture_duration - elapsed:.1f}s", True, (255, 255, 255))
                scr.blit(text, (10, 10))

                while not data_queue.empty():
                    emg = data_queue.get()
                    recorded_emg.append(emg)

                    gesture_real_label = None    # Перевод целочисленный названий жестов в порядковые 
                    if '#0' in gesture_name:
                        gesture_real_label = 0
                    else:
                        gesture_real_label = gesture_dict_labels[gesture_name]

                    gesture_labels.append(gesture_real_label)
                    plot(scr, [e / PLOT_REDUCTION for e in emg])

                pygame.display.flip()

            clock.tick(200)

    except KeyboardInterrupt:
        print("Interrupted")

    finally:
        cmd_queue.put('stop')
        p_worker.join()
        pygame.quit()
        sys.exit()
