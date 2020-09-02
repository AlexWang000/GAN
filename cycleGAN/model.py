# G: 2downsample + 9ResNet + 2upsample
# D: C64 + C128 + C256 + C512 + flat
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, LeakyReLU, add

class ReflectionPad(keras.layers.Layer):
    def __init__(self, k):
        super(ReflectionPad, self).__init__()
        self.pad = [[0, 0], [k, k], [k, k], [0, 0]]
    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode='REFLECT')


class ResNetBlock(keras.layers.Layer):
    def __init__(self):
        super(ResNetBlock, self).__init__()
        self.layers = [
            ReflectionPad(1),
            Conv2D(256, 3, use_bias=False),
            InstanceNormalization(),
            ReLU(),

            ReflectionPad(1),
            Conv2D(256, 3, use_bias=False),
            InstanceNormalization()
        ]
    def call(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return add([inputs, outputs])


def generator():
    layers = [
        keras.Input((256,256, 3)),

        ReflectionPad(3),
        Conv2D(64,7,use_bias=False),
        InstanceNormalization(),
        ReLU(),

        Conv2D(128,3,2,'same', use_bias=False),
        InstanceNormalization(),
        ReLU(),

        Conv2D(256,3,2,'same', use_bias=False),
        InstanceNormalization(),
        ReLU()
    ]

    layers += [ResNetBlock() for i in range(9)]

    layers += [
        Conv2DTranspose(128, 3, 2, 'same', use_bias=False),
        InstanceNormalization(),
        ReLU(),

        Conv2DTranspose(64, 3, 2, 'same', use_bias=False),
        InstanceNormalization(),
        ReLU(),

        ReflectionPad(3),
        Conv2D(3, 7, activation='tanh'),
    ]
    return keras.Sequential(layers)


def discriminator():
    layers = [
        keras.Input((256, 256, 3)),
        Conv2D(64, 4, 2, 'same'),
        LeakyReLU(alpha=0.2),

        Conv2D(128, 4, 2, 'same', use_bias=False),
        InstanceNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(256, 4, 2, 'same', use_bias=False),
        InstanceNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(512, 4, 1, 'same', use_bias=False),
        InstanceNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(1, 4, 1, 'same')
    ]
    return keras.Sequential(layers)
