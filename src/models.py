from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
from config import *


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )

    def call(self, inputs):
        # Пространственное внимание
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)  # [batch, H, W, 2]
        mask = self.conv(concat)                           # [batch, H, W, 1]
        return inputs * mask                               # Элементное умножение
    

def spatial_attention(input_feature: np.ndarray):
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat_pool = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])

    CNN_POOL = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', 
                                      kernel_initializer='he_normal', use_bias=False)
    spatial_mask = CNN_POOL(concat_pool)

    return tf.keras.layers.Multiply()([input_feature, spatial_mask])


def build_autoencoder(input_shape: tuple=INPUT_SHAPE_BASE, filters: tuple=FILTERS_BASE, kernel_size: tuple=KERNEL_SIZE_BASE, pool_size: tuple=POOL_SIZE_BASE):
    pass

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
                                  kernel_size=kernel_size, activation='relu', padding='same')
    
    CNN2 = tf.keras.layers.Conv2D(filters=filters[1], strides=1,
                                  kernel_size=kernel_size, activation='relu', padding='same')
    
    model = tf.keras.Sequential([
        
        tf.keras.layers.Input(shape=input_shape),
        CNN1, 
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


def build_SAM_model(input_shape: tuple=INPUT_SHAPE_BASE, filters: tuple=FILTERS_BASE, kernel_size: tuple=KERNEL_SIZE_BASE, pool_size: tuple=POOL_SIZE_BASE, 
                    p_dropout: float=P_DROPOUT_BASE, num_classes: int=NUM_CLASSES) -> tf.keras.Sequential:
    
    CNN1 = tf.keras.layers.Conv2D(filters=filters[0], strides=1,
                                  kernel_size=kernel_size, activation='relu', padding='same')
    
    CNN2 = tf.keras.layers.Conv2D(filters=filters[1], strides=1,
                                  kernel_size=kernel_size, activation='relu', padding='same')
    
    inputs = tf.keras.Input(shape=input_shape)
    x = CNN1(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.SpatialDropout2D(rate=p_dropout)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding='same')(x)

    x = CNN2(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = SpatialAttention()(x)    # NOTE: SAM должен быть здесь, т.к. это детерменированный блок
    x = tf.keras.layers.BatchNormalization()(x)
    # x = spatial_attention(x)    
    x = tf.keras.layers.SpatialDropout2D(rate=p_dropout)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding='same')(x)

    # NOTE: Классификатор
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    outputs = tf.keras.layers.Softmax(axis=-1)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def main():
    pass


if __name__ == "__main__":
    main()