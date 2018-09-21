import math

import tensorflow as tf


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x * 1.0))


if __name__ == '__main__':
    a = [[1, 2, 3], [3, 4, 5]]
    a = tf.constant(a)
    sess = tf.Session()
    loc = sess.run(tf.where(tf.greater(a, 2)))
    a[loc] = 10
    print(sess.run(a))
