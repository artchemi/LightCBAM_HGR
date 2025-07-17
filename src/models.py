from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
from config import *


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        """Пространственный модуль внимания

        Args:
            kernel_size (int, optional): Размер ядра. Defaults to 3.
        """
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
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)  # [batch, H, W, 2]
        mask = self.conv(concat)                           # [batch, H, W, 1]
        return inputs * mask    # Элементное умножение


class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=4, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        _, time_steps, channels = input_shape
        reduced_channels = max(1, channels // self.reduction_ratio)

        self.shared_dense_one = tf.keras.layers.Dense(
            units=reduced_channels,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True
        )
        self.shared_dense_two = tf.keras.layers.Dense(
            units=channels,
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=True
        )
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs, return_attention=False):
        # inputs: [batch, time, channels]
        avg_pool = tf.reduce_mean(inputs, axis=1)
        max_pool = tf.reduce_max(inputs, axis=1)

        avg_out = self.shared_dense_one(avg_pool)
        avg_out = self.shared_dense_two(avg_out)

        max_out = self.shared_dense_one(max_pool)
        max_out = self.shared_dense_two(max_out)

        attention = tf.nn.sigmoid(avg_out + max_out)  # [batch, channels]
        attention = tf.expand_dims(attention, axis=1)  # [batch, 1, channels]

        output = inputs * attention  # broadcasting
        if return_attention:
            return output, attention
        else:
            return output

# Пример использования при 8 входных каналах:
# model.add(ChannelAttention(reduction_ratio=4))


def build_base_model(input_shape: tuple=INPUT_SHAPE_BASE, filters: tuple=FILTERS_BASE, kernel_size: tuple=KERNEL_SIZE_BASE, 
                     pool_size: tuple=POOL_SIZE_BASE, p_dropout: float=P_DROPOUT_BASE, num_classes: int=NUM_CLASSES) -> tf.keras.Sequential:
    """Базовая сверточная модель MIC-Laboratory/IEEE-NER-2023-EffiE.

    Args:
        input_shape (tuple, optional): Размерность входа. По умолчанию (W, H, 1), где W - ширина окна, H - количество каналов. Defaults to INPUT_SHAPE_BASE.
        filters (tuple, optional): Размерности фильтров. Defaults to FILTERS_BASE.
        kernel_size (tuple, optional): Размерность ядер свертки. Defaults to KERNEL_SIZE_BASE.
        pool_size (tuple, optional): Размерность пулинга. Defaults to POOL_SIZE_BASE.
        p_dropout (float, optional): Коэффициент дропаута. Defaults to P_DROPOUT_BASE.
        num_classes (int, optional): Количество жестов/классов. Defaults to NUM_CLASSES.

    Returns:
        tf.keras.Sequentia: Модель TF.
    """
    CNN1 = tf.keras.layers.Conv2D(filters=filters[0], strides=1, kernel_size=kernel_size, padding='same')
    
    CNN2 = tf.keras.layers.Conv2D(filters=filters[1], strides=1, kernel_size=kernel_size, padding='same')
    
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
    
    # Класификатор    NOTE: Можно попробовать заменить потом на GPFlow
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Softmax(axis=-1))

    return model


def build_SAM_model(input_shape: tuple=INPUT_SHAPE_BASE, filters: tuple=FILTERS_BASE, kernel_size: tuple=KERNEL_SIZE_BASE, 
                    pool_size: tuple=POOL_SIZE_BASE, p_dropout: float=P_DROPOUT_BASE, num_classes: int=NUM_CLASSES) -> tf.keras.Sequential:
    """Модель с пространственным механизмом внимания.

    Args:
        input_shape (tuple, optional): Размерность входа. По умолчанию (W, H, 1), где W - ширина окна, H - количество каналов. Defaults to INPUT_SHAPE_BASE.
        filters (tuple, optional): Размерности фильтров. Defaults to FILTERS_BASE.
        kernel_size (tuple, optional): Размерность ядер свертки. Defaults to KERNEL_SIZE_BASE.
        pool_size (tuple, optional): Размерность пулинга. Defaults to POOL_SIZE_BASE.
        p_dropout (float, optional): Коэффициент дропаута. Defaults to P_DROPOUT_BASE.
        num_classes (int, optional): Количество жестов/классов. Defaults to NUM_CLASSES.

    Returns:
        tf.keras.Sequential: Модель TF.
    """
    
    CNN1 = tf.keras.layers.Conv2D(filters=filters[0], strides=1,
                                  kernel_size=kernel_size, padding='same')
    
    CNN2 = tf.keras.layers.Conv2D(filters=filters[1], strides=1,
                                  kernel_size=kernel_size, padding='same')
    
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
    x = tf.keras.layers.SpatialDropout2D(rate=p_dropout)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding='same')(x)

    # Классификатор
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    outputs = tf.keras.layers.Softmax(axis=-1)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_CAM_model_1D(input_shape: tuple, filters: tuple=FILTERS_BASE, kernel_size: int=KERNEL_SIZE_BASE_1D,pool_size: int=POOL_SIZE_BASE_1D,
                       p_dropout: float=P_DROPOUT_BASE,num_classes: int=NUM_CLASSES, return_attention_mask: bool = False) -> tf.keras.Model:
    
    inputs = tf.keras.Input(shape=input_shape)  # (T, C)

    # Канальное внимание на входе
    ca_layer = ChannelAttention(reduction_ratio=max(1, input_shape[-1] // 4))
    
    if return_attention_mask:
        x, attention = ca_layer(inputs, return_attention=True)
    else:
        x = ca_layer(inputs)

    # Первый 1D‑свёрточный блок
    x = tf.keras.layers.Conv1D(filters=filters[0], kernel_size=kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.SpatialDropout1D(rate=p_dropout)(x)
    x = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')(x)

    # Второй 1D‑свёрточный блок
    x = tf.keras.layers.Conv1D(filters=filters[1], kernel_size=kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.SpatialDropout1D(rate=p_dropout)(x)
    x = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')(x)

    # Классификатор
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    outputs = tf.keras.layers.Softmax(axis=-1)(x)

    if return_attention_mask:
        return tf.keras.Model(inputs=inputs, outputs=[outputs, attention])
    else:
        return tf.keras.Model(inputs=inputs, outputs=outputs)


# Пример настройки констант:
# INPUT_SHAPE_BASE_1D = (56, 8)            # 56 отсчётов, 8 каналов
# KERNEL_SIZE_BASE_1D = 3                  # ширина ядра 3
# POOL_SIZE_BASE_1D = 2                    # пулинг длиной 2
# FILTERS_BASE = (64, 64)                  # два блока по 64 фильтра
# P_DROPOUT_BASE = 0.5
# NUM_CLASSES = 10

def main():
    pass


if __name__ == "__main__":
    main()