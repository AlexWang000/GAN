import tensorflow as tf
from .loss import generatorLoss, discriminatorLoss
from .model import generator, discriminator
from .load import load, manage, show, saveSample

# 40 epoches and learning rate will be constant 0.0002
# because its pretty imfesible to train few hundred epochs as a poor student
# the pooling from original paper is left out because seems like it slows down training

epochs = 40

@tf.function
def step(G, F, DX, DY, X, Y, gOpt, dOpt):
    with tf.GradientTape(persistent=True) as tape:
        fakeY = G(X, training=True)
        cycleX = F(fakeY, training=True)

        fakeX = F(Y, training=True)
        cycleY = G(fakeX, training=True)

        identityX = F(X, training=True)
        identityY = G(Y, training=True)

        predFakeX = DX(fakeX, training=True)
        predFakeY = DY(fakeY, training=True)

        predX = DX(X, training=True)
        predY = DY(Y, training=True)

        gLoss = generatorLoss(X, Y, predFakeX, predFakeY, cycleX, cycleY, identityX, identityY)
        dLoss = discriminatorLoss(predX, predFakeX, predY, predFakeY)

    gVars = G.trainable_variables + F.trainable_variables
    dVars = DX.trainable_variables + DY.trainable_variables
    gGrad = tape.gradient(gLoss, gVars)
    dGrad = tape.gradient(dLoss, dVars)
    gOpt.apply_gradients(zip(gGrad, gVars))
    dOpt.apply_gradients(zip(dGrad, dVars))
    return (gLoss, dLoss)


def train(visualize=False):
    trainX, trainY, testX, testY = load()
    G, F = generator(), generator()
    DX, DY = discriminator(), discriminator()
    gOpt, dOpt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    data = tf.data.Dataset.zip((trainX, trainY))
    length = len(data)
    manager = manage(G, F, DX, DY, gOpt, dOpt)
    sampleX, sampleY = None, None
    for epoch in range(epochs):
        i = 0
        for X, Y in data:
            if i == 0:
                if sampleX is None:
                    sampleX, sampleY = X, Y
                saveSample(G, F, sampleX, sampleY, epoch)
            gl, dl = step(G, F, DX, DY, X, Y, gOpt, dOpt)
            i += 1
            if i % 100 == 0:
                print(f"{i}/{length} iteration of {epoch}/{epochs} epoch: generator loss: ${gl}, discriminator loss: ${dl}")
                if visualize:
                    show(G, F, X, Y)
        manager.save()
            
        
