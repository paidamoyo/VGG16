import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

flags = {
    'hidden_size': 10
}


def _cost_svm(x, y, w, b, C):
    f = tf.matmul(x, w)
    f = tf.add(f, b)
    l = tf.maximum(tf.constant(0, dtype=tf.float32), 1 - tf.mul(y, f))
    return tf.squeeze(0.5 * tf.reduce_sum(tf.square(w)) + tf.mul(C, l))

def _set_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

def _create_data(num_points):
    x0 = np.random.normal(0, 0.3, [num_points, 2])
    y0 = -1 * np.ones(num_points)
    x1 = np.random.normal(1.5, 0.3, [num_points, 2])
    y1 = np.ones(num_points)
    data_x = np.concatenate((x0, x1), axis=0)
    data_y = np.concatenate((y0, y1), axis=0)
    return x0, x1, data_x, data_y

def main():
    _set_seed(1234)
    restore = True
    x0, x1, data_x, data_y = _create_data(500)
    C = tf.constant([1], dtype=tf.float32)  #  tf.Variable(tf.constant([1], dtype=tf.float32), dtype=tf.float32)
    w = tf.Variable(initial_value=tf.random_uniform([2, 1]), dtype=tf.float32)
    b = tf.Variable(tf.constant([0], dtype=tf.float32), dtype=tf.float32)

    x = tf.placeholder(shape=[1, 2], dtype=tf.float32)
    y = tf.placeholder(shape=[1, ], dtype=tf.float32)

    cost = _cost_svm(x, y, w, b, C)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    tf.histogram_summary("C", C)
    tf.histogram_summary("w", w)
    tf.histogram_summary("b", b)
    tf.scalar_summary("cost", cost)

    sess = tf.InteractiveSession()
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('./aux', sess.graph)
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()


    if restore is True:
        saver.restore(sess, './aux/starting.ckpt')
    else:
        sess.run(init)
        for step in range(5000):
            ind = np.random.randint(low=0, high=1000, size=1)
            summary, _ = sess.run([merged, optimizer], feed_dict={x: data_x[ind, :], y: data_y[ind]})
            writer.add_summary(summary=summary, global_step=step)
            if step % 100 == 0:
                print(step)
        save_path = saver.save(sess, './aux/starting.ckpt')

    w_plot, b_plot = sess.run([w, b])
    print(w_plot)
    print(b_plot)
    x_plot = np.linspace(-1, 3.5)
    y_plot = -(np.dot(w_plot[0, ], np.expand_dims(x_plot, 0)) + b_plot) / w_plot[1, ]
    y_neg1 = -(np.dot(w_plot[0, ], np.expand_dims(x_plot, 0)) + b_plot + 1) / w_plot[1, ]
    y_pos1 = -(np.dot(w_plot[0, ], np.expand_dims(x_plot, 0)) + b_plot - 1) / w_plot[1, ]
    plt.scatter(x0[:, 0], x0[:, 1])
    plt.scatter(x1[:, 0], x1[:, 1], color='g')
    plt.plot(x_plot, y_plot, color='r')
    plt.plot(x_plot, y_neg1, color='k')
    plt.plot(x_plot, y_pos1, color='k')
    plt.show()


if __name__ == '__main__':
    main()


