import tensorflow as tf
import re
import h5py

# Create some wrappers for simplicity
def conv2d(img, w, b, strides=1):
    img = tf.nn.conv2d(img, w, strides=[1, strides, strides, 1], padding='SAME')
    img = tf.nn.bias_add(img, b)
    return tf.nn.relu(img)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def vgg16(x, weights, biases):
    block1_conv1 = conv2d(x, weights['b1_c1'], biases['b1_c1'])
    block1_conv2 = conv2d(block1_conv1, weights['b1_c2'], biases['b1_c2'])
    block1_pool = maxpool2d(block1_conv2, k=2)

    block2_conv1 = conv2d(block1_pool, weights['b2_c1'], biases['b2_c1'])
    block2_conv2 = conv2d(block2_conv1, weights['b2_c2'], biases['b2_c2'])
    block2_pool = maxpool2d(block2_conv2, k=2)

    block3_conv1 = conv2d(block2_pool, weights['b3_c1'], biases['b3_c1'])
    block3_conv2 = conv2d(block3_conv1, weights['b3_c2'], biases['b3_c2'])
    block3_conv3 = conv2d(block3_conv2, weights['b3_c3'], biases['b3_c3'])
    block3_pool = maxpool2d(block3_conv3, k=2)

    block4_conv1 = conv2d(block3_pool, weights['b4_c1'], biases['b4_c1'])
    block4_conv2 = conv2d(block4_conv1, weights['b4_c2'], biases['b4_c2'])
    block4_conv3 = conv2d(block4_conv2, weights['b4_c3'], biases['b4_c3'])
    block4_pool = maxpool2d(block4_conv3, k=2)

    block5_conv1 = conv2d(block4_pool, weights['b5_c1'], biases['b5_c1'])
    block5_conv2 = conv2d(block5_conv1, weights['b5_c2'], biases['b5_c2'])
    block5_conv3 = conv2d(block5_conv2, weights['b5_c3'], biases['b5_c3'])
    return block5_conv3


def spp_layer(mapstack, dims, poolnum):
    maps = tf.unpack(mapstack, axis=3)
    flattened = []
    for mapnum in range(len(maps)):
        for p in poolnum:
            div = tf.constant(p, dtype=tf.int32)
            size = tf.reshape(
                tf.pack([tf.constant([0], dtype=tf.int32), tf.to_int32(dims[0] / div), tf.to_int32(dims[1] / div)]),
                [3, ])
            for i in range(p):
                i = tf.constant(i, dtype=tf.int32)
                for j in range(p):
                    j = tf.constant(j, dtype=tf.int32)
                    x = tf.to_int32(dims[0] / div)
                    y = tf.to_int32(dims[0] / div)
                    begin = tf.reshape(
                        tf.pack([tf.constant([0], dtype=tf.int32), tf.to_int32(i * x), tf.to_int32(j * y)]), [3, ])
                    flattened.append(tf.reduce_max(tf.slice(maps[0], begin=begin, size=size)))
    return tf.reshape(tf.convert_to_tensor(flattened), [1,2560])

# 1536 (12) x 3264 (12) = 128 (16) x 272 (16) = 8 *17
def max_flatten(mapstack, params):
    pool = tf.nn.max_pool(mapstack, ksize=[1, 17, 8, 1], strides=[1, 17, 8, 1], padding='SAME')
    mapmax = tf.reshape(pool,[params['batch_size'], 512])
    flatten = tf.reduce_max(mapmax, reduction_indices=[0])
    return tf.reshape(flatten, [1, 512])


def fc_layer(flattened, weights, biases, dropout):
    fc1 = tf.add(tf.matmul(flattened, weights['fc1']), biases['fc1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    output = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return output


def define_parameters(flags):
    weights['fc1'] = tf.Variable(tf.random_normal([512, 256]))
    weights['out'] = tf.Variable(tf.random_normal([256, 2]))
    biases['fc1'] = tf.Variable(tf.random_normal([256]))
    biases['out'] = tf.Variable(tf.random_normal([2]))
    return weights, biases

def load_pretrained_parameters_VGG(flags):
    filename = flags['code_directory'] + 'aux/vgg16_ImageNet_tf.h5'
    pretrained = h5py.File(filename, 'r')
    weights = {}
    biases = {}
    for layer_key in list(pretrained.keys()): #'block' and 'conv' to define layer
        m = re.search('block(.+?)_conv(.+?)', layer_key)
        if m is not None:
            for nums_key in list(pretrained[layer_key].keys()):
                exp = re.search('block' + m.group(1) + '_conv' + m.group(2) + '_(.+?)', nums_key)
                if exp.group(1) == 'W':
                    weights['b' + m.group(1) + '_c' + m.group(2)] = \
                        tf.Variable(pretrained[layer_key][nums_key].value, trainable=False)
                if exp.group(1) == 'b':
                    biases['b' + m.group(1) + '_c' + m.group(2)] = \
                        tf.Variable(pretrained[layer_key][nums_key].value, trainable=False)
                else:
                    ValueError('Could not locate weights in pretrained dictionary!')
        else:
            continue  # skip fc weights and pool layers
    return weights, biases


def build_model(x, flags, params):
    weights, biases = load_pretrained_parameters_VGG(flags)
    mapstack = vgg16(x=x, weights=weights, biases=biases)
    flat = max_flatten(mapstack, params)
    output = fc_layer(flattened=flat, weights=weights, biases=biases, dropout=params['dropout'])
    return output

def compute_VGG(x, flags):
    weights, biases = load_pretrained_parameters_VGG(flags)
    mapstack = vgg16(x=x, weights=weights, biases=biases)
    return mapstack