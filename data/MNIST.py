import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def load_data_MNIST():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist

def generate_MNIST(mnist, batch_size):
    batch_x, labels = mnist.train.next_batch(batch_size)
    batch_x = np.reshape(batch_x, [batch_size, 28, 28, 1])
    return labels, batch_x