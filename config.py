# NOTE: Данные для обучения
FOLDAR_PATH = "Ninapro_DB5"
STD_MEAN_PATH = "scaling_params.json"

EXERCISES = ["E2"]
MYO_PREF = "elbow"

# Our: [0, 13, 15, 17, 18, 19, 20, 25, 26]
# MIC-Laboratory/IEEE-NER-2023-EffiE: [0, 13, 15, 17, 18, 25, 26]
GESTURE_INDEXES = [0, 13, 15, 17, 18, 25, 26]
NUM_CLASSES = len(GESTURE_INDEXES)

VALID_SIZE = 0.5    # Доля от test
TEST_SIZE = 0.3    # Доля от общей выборки
BATCH_SIZE = 384    # ???

INIT_LR= 0.2

WINDOW_SIZE = 32
STEP_SIZE = 16

EPOCHS = 200
PATIENCE = 50

# NOTE: Параметры базовой CNN
INPUT_SHAPE_BASE = (WINDOW_SIZE, 8, 1)
FILTERS_BASE = (48, 96)
KERNEL_SIZE_BASE = (3, 3)
POOL_SIZE_BASE = (1, 2)
P_DROPOUT_BASE = 0.5

# NOTE: Прочее
GLOBAL_SEED = 42
SAVE_PATH = "checkpoints/model.ckpt"