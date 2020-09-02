import tensorflow as tf
from .loss import *
from .model import *

epochs = 40

@tf.funtion
def step(G, D, X, gLossFunc, dLossFunc, gOpt, dOpt):
    noise = tf.random.normal([8, 100])
    with tf.GradientTape(persistent=True) as tape:
        fake = G(noise, training=True)
        pred = D(X, training=True)
        predFake = D(fake, training=True)

        gLoss = gLossFunc(predFake)
        dLoss = dLossFunc(pred, predFake)
    gGrad = tape.gradient(gLoss, G.trainable_variables)
    dGrad = tape.gradient(dLoss, D.trainable_variables)
    gOpt.apply_gradients(zip(gGrad, G.trainable_variables))
    dOpt.apply_gradients(zip(dGrad, D.trainable_variables))


def train(dataset, type):
    G = generator()
    D = discriminator()
    gOpt, dOpt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    for epoch in epochs:
        for X in dataset:
            step(G, D, X, DCGANGLoss, DCGANDLoss, gOpt, dOpt)


