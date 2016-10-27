import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def load_data_MNIST():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist

def generate_MNIST(mnist, batch_size):
    batch_x, labels = mnist.train.next_batch(batch_size)
    batch_x = np.reshape(batch_x[0], [batch_size, 28, 28, 1])
    print(batch_x.shape)
    print(labels.shape)
    return labels, batch_x