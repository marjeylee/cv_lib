# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     __init__.py
   Description :
   Author :       'li'
   date：          2018/9/13
-------------------------------------------------
   Change Activity:
                   2018/9/13:
-------------------------------------------------
"""
# __author__ = 'li'
# import tensorflow as tf
#
# q = tf.ones((256, 256), dtype=tf.float32) * -1
# p = tf.ones((256, 256), dtype=tf.float32) * 555555
# # eps = 1e-7
# # p = tf.clip_by_value(p, eps, 1.0 - eps)
# # a = tf.multiply(q, tf.log(p))
# # b = tf.multiply((1 - q), tf.log(1 - p))
# cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=q, logits=p)
# cross_entropy = tf.reduce_sum(cross_entropy)
#
# sess = tf.Session()
# result = sess.run(cross_entropy)
# print(result)
# import numpy as  np
#
# a = [1, 2, 3, 4, 5, 5, 6]
# a = np.array(a)
# b = np.where(a > 4)
# print(a[b])
