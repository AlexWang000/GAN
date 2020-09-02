import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Reshape, LeakyReLU, Flatten


def generator():
    layers = [
        keras.Input((100,)),
        Dense(4*4*1024, use_bias = False),
        BatchNormalization(),
        ReLU(),

        Reshape((4,4,1024)),

        Conv2DTranspose(512, 5, 2, 'same', use_bias=False),
        BatchNormalization(),
        ReLU(),
        Conv2DTranspose(256, 5, 2, 'same', use_bias=False),
        BatchNormalization(),
        ReLU(),
        Conv2DTranspose(128, 5, 2, 'same', use_bias=False),
        BatchNormalization(),
        ReLU(),

        Conv2DTranspose(3, 5, 2, 'same', use_bias=False, activation="tanh")
    ]
    return keras.Sequential(layers)


def discriminator():
    layers = [
        keras.Input((64, 64, 3)),

        Conv2D(64, 4, 2, 'same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2D(128, 4, 2, 'same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2D(256, 4, 2, 'same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Flatten(),
        Dense(1, activation='sigmoid')
    ]
    return keras.Sequential(layers)


