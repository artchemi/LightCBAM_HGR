from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
from config import *


def build_base_model(input_shape: tuple=INPUT_SHAPE_BASE, filters: tuple=FILTERS_BASE, kernel_size: tuple=KERNEL_SIZE_BASE, pool_size: tuple=POOL_SIZE_BASE, 
                     p_dropout: float=P_DROPOUT_BASE, num_classes: int=NUM_CLASSES) -> tf.keras.Sequential:
    """Базовая сверточная модель MIC-Laboratory/IEEE-NER-2023-EffiE.

    Args:
        input_shape (tuple, optional): Размерность входа. По умолчанию (W, H, 1), где W - ширина окна, H - количество каналов. Defaults to INPUT_SHAPE_BASE.
        filters (tuple, optional): Размерности фильтров. Defaults to FILTERS_BASE.
        kernel_size (tuple, optional): Размерность ядер свертки. Defaults to KERNEL_SIZE_BASE.
        pool_size (tuple, optional): Размерность пулинга. Defaults to POOL_SIZE_BASE.
        p_dropout (float, optional): Коэффициент дропаута. Defaults to P_DROPOUT_BASE.
        num_classes (int, optional): Количество жестов/классов. Defaults to NUM_CLASSES.

    Returns:
        _type_: _description_
    """
    CNN1 = tf.keras.layers.Conv2D(filters=filters[0], strides=1,
                                  kernel_size=kernel_size, activation='relu')
    
    CNN2 = tf.keras.layers.Conv2D(filters=filters[1], strides=1,
                                  kernel_size=kernel_size, activation='relu')
    
    model = tf.keras.Sequential([
        
        tf.keras.layers.Input(shape=input_shape), CNN1, 
        tf.keras.layers.BatchNormalization(), tf.keras.layers.PReLU(),
        tf.keras.layers.SpatialDropout2D(rate=p_dropout),
        tf.keras.layers.MaxPool2D(pool_size=pool_size, padding='same'),

        CNN2, 
        tf.keras.layers.BatchNormalization(), tf.keras.layers.PReLU(),
        tf.keras.layers.SpatialDropout2D(rate=p_dropout),
        tf.keras.layers.MaxPool2D(pool_size=pool_size, padding='same'),
        
        tf.keras.layers.Flatten()
        ])
    
    # NOTE: Класификатор
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Softmax(axis=-1))

    return model

def build_SAM_model(input_shape: tuple, num_classes: int=8):
    pass

def build_CAM_model(input_shape: tuple, num_classes: int=8):
    pass

def build_CBAM_model(input_shape: tuple, num_classes: int=8):
    pass


def main():
    model = build_base_model()
    print(model)


if __name__ == "__main__":
    main()