from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def load_data():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist

def generate_MNIST(mnist, batch_size):
    return mnist.train.next_batch(batch_size)