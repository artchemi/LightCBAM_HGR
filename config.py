# NOTE: Данные для обучения
FOLDER_PATH = "Ninapro_DB5"
STD_MEAN_PATH = "scaling_params.json"

EXERCISES = ["E2", "E3"]
MYO_PREF = "elbow"

# Our: [0, 13, 15, 17, 18, 19, 20, 25, 26]
# MIC-Laboratory/IEEE-NER-2023-EffiE: [0, 13, 15, 17, 18, 25, 26]
GESTURE_INDEXES_B = [0, 13, 15, 18, 19]
GESTURE_INDEXES_C = [34, 38, 43, 46]
GESTURE_INDEXES_MAIN = GESTURE_INDEXES_B + GESTURE_INDEXES_C
CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7]    # NOTE: CAM-MS-RS - [0, 3, 4, 5, 6]
NUM_CLASSES = len(GESTURE_INDEXES_MAIN)

VALID_SIZE = 0.5    # Доля от test
TEST_SIZE = 0.3    # Доля от общей выборки
BATCH_SIZE = 384    # ???

INIT_LR= 0.2

WINDOW_SIZE = 56
STEP_SIZE = 16

EPOCHS = 200
PATIENCE = 50

# NOTE: Параметры базовой CNN
INPUT_SHAPE_BASE = (WINDOW_SIZE, len(CHANNELS), 1)
FILTERS_BASE = (48, 96)
KERNEL_SIZE_BASE = (1, 3)    # (3, 3)
POOL_SIZE_BASE = (1, 2)
P_DROPOUT_BASE = 0.5

# NOTE: Прочее
GLOBAL_SEED = 42
SAVE_PATH = "checkpoints/model"