# this preprocess functions come from tensorflow
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASETNAME = 'cycle_gan/horse2zebra'

def load():
    def random_crop(image):
        cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
        return cropped_image

    def normalize(image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image

    def random_jitter(image):
        image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = random_crop(image)
        image = tf.image.random_flip_left_right(image)
        return image

    def preprocess_image_train(image, label):
        image = random_jitter(image)
        image = normalize(image)
        return image

    def preprocess_image_test(image, label):
        image = normalize(image)
        return image

    dataset, metadata = tfds.load(DATASETNAME, with_info=True, as_supervised=True)
    trainA, trainB = dataset['trainA'], dataset['trainB']
    testA, testB = dataset['testA'], dataset['testB']

    trainA = trainA.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
    trainB = trainB.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
    testA = testA.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
    testB = testB.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
    return (trainA, trainB, testA, testB)


def manage(G, F, DX, DY, gOpt, dOpt):
    path = "./checkpoint"
    checkpoint = tf.train.Checkpoint(G=G, F=F, DX=DX, DY=DY, gOpt=gOpt, dOpt=dOpt)
    return tf.train.CheckpointManager(checkpoint, path, 5)

def show(G, F, X, Y):
    fakeY = G(X, training=False)
    fakeX = F(Y, training=False)
    cycleX = F(fakeY, training=False)
    cycleY = G(fakeX, training=False)

    imgs = [X[0], fakeY[0], cycleX[0], Y[0], fakeX[0], cycleY[0]]
    titles = ["X", "fakeY", "cycleX", "Y", "fakeX", "cycleY"]

    plt.figure(figsize=(8, 8))
    for i in range(len(imgs)):
        plt.subplot(2, 3, i+1)
        plt.title(titles[i])
        plt.imshow(imgs[i])
    plt.show()

def saveSample(G, F, X, Y, epoch):
    fakeY = G(X, training=False)
    fakeX = F(Y, training=False)
    imgs = [X[0], fakeY[0], Y[0], fakeX[0]]
    titles = ["X", "fakeY", "Y", "fakeX"]
    plt.figure(figsize=(20, 20))
    for i in range(len(imgs)):
        plt.subplot(2, 3, i+1)
        plt.title(titles[i])
        plt.imshow(imgs[i])
    plt.savefig(f"samples/{epoch}.png")
