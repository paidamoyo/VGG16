import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def load_data_MNIST():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist

def generate_MNIST(mnist, batch_size):
    return tf.reshape(mnist.train.next_batch(batch_size), [batch_size, 28, 28, 1])