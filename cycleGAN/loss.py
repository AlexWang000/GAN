import tensorflow as tf

adversarialLossWeight = 1
cycleLossWeight = 10
identityLossWeight = 0
MSE = tf.keras.losses.MeanSquaredError()
MAE = tf.keras.losses.MeanAbsoluteError()

def adversarialLoss(pred):
    return MSE(tf.ones_like(pred), pred)
def cycleLoss(truth, pred):
    return MAE(truth, pred)
def identityLoss(truth, pred):
    return MAE(truth, pred)

def generatorLoss(X, Y, predFakeX, predFakeY, cycleX, cycleY, identityX, identityY):
    al = adversarialLoss(predFakeX) + adversarialLoss(predFakeY)
    cl = cycleLoss(X, cycleX) + cycleLoss(Y, cycleY)
    il = identityLoss(X, identityX) + identityLoss(Y, identityY)
    return al * adversarialLossWeight + cl * cycleLossWeight + il * identityLossWeight

def discriminatorLoss(predX, predY, predFakeX, predFakeY):
    l = MSE(tf.ones_like(predX), predX) + MSE(tf.zeros_like(predFakeX), predFakeX) + \
        MSE(tf.ones_like(predY), predY) + MSE(tf.zeros_like(predFakeY), predFakeY)
    return l


