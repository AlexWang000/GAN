import tensorflow as tf
CE = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def DCGANGLoss(pred):
    return CE(tf.ones_like(pred), pred)

def DCGANDLoss(truth, pred):
    return CE(tf.ones_like(truth), truth) + CE(tf.zeros_like(pred), pred)
