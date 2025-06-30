import os
import random
import numpy as np
import tensorflow as tf


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
    