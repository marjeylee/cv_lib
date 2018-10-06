# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train
   Description :
   Author :       'li'
   date：          2018/9/14
-------------------------------------------------
   Change Activity:
                   2018/9/14:
-------------------------------------------------
"""
import time

from app import log_util
from app.text_area_mask.bulid_model import MaskModel
import tensorflow as tf

from app.text_area_mask.generate_data import load_batch_training_data  # , add_training_data
import numpy as np


def get_loss(mask_model):
    output = mask_model.output
    output_shape = output.shape.dims[3].value
    output = mask_model.conv_layer(output, 1, output_shape, 1, 1, name='tmp_output')
    label = tf.placeholder(dtype=tf.float32, shape=(None, None, None))

    resize_output = tf.image.resize_bilinear(output, size=[tf.shape(label)[1], tf.shape(label)[2]])
    output = tf.squeeze(resize_output, axis=3)
    p = tf.sigmoid(output)
    q = label
    # eps = 1e-7
    # p = tf.clip_by_value(p, eps, 1.0 - eps)
    # a = tf.multiply(q, tf.log(p))
    # b = tf.multiply((1 - q), tf.log(1 - p))
    # cross_entropy = -tf.add(a, b)

    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output)
    # loss = tf.reduce_sum(cross_entropy) * 10000

    eps = 1e-5
    intersection = tf.reduce_sum(q * p)
    union = tf.reduce_sum(q) + tf.reduce_sum(p) + eps
    loss = 1. - (2 * intersection / union)
    loss = loss * 10000000000000
    return label, loss, output


def train():
    log_util.info('building model')
    mask_model = MaskModel()
    mask_model.build()
    input_images = mask_model.input_images
    label, loss, predict_output = get_loss(mask_model)
    opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto()) as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)
        saver.restore(sess=sess, save_path='./model/model')
        batch_size = 22
        training_data = load_batch_training_data(batch_size)
        feed_dict = {input_images: training_data[0], label: training_data[1]}
        satisfy = 0
        start_time = time.time()
        end_time = None
        for i in range(999999999999999):
            if i % 2 == 0:
                training_data = load_batch_training_data(batch_size)
                feed_dict = {input_images: training_data[0], label: training_data[1]}
            if i % 10 == 0:
                end_time = time.time()
                during_time = end_time - start_time
                log_util.info('batch_size : ' + str(batch_size) + ' , ' + 'cost_time :' + str(during_time / 1000.0))
                start_time = time.time()
                loss_result = sess.run(loss, feed_dict=feed_dict)
                print('current step : ' + str(i) + ',loss : ' + str(loss_result))
                opt.run(feed_dict=feed_dict)
                loss_result = sess.run(loss, feed_dict=feed_dict)
                print('current step : ' + str(i) + ',loss : ' + str(loss_result))
                # if loss_result / 100000000000 < 0.02:
                #     satisfy = satisfy + 1
                #     log_util.info('satisfy : ' + str(satisfy))
                #     if satisfy > 5:
                #         satisfy = 0
                #         add_training_data(batch_size=batch_size)
                # else:
                #     satisfy = 0
                # print(sess.run(label, feed_dict=feed_dict))
                # if i % 200 == 0:
                #     out_result = sess.run(predict_output, feed_dict=feed_dict)
                #     print(out_result)
                # b = np.where(out_result > 4)
                # print(b)
                # print(out_result[b])
            # print('current step : ' + str(i) + ',loss : ' + str(loss_result))
            # label_result = sess.run(label, feed_dict=feed_dict)
            # conv1_relu_result = sess.run(conv1_relu, feed_dict=feed_dict)
            # predict_output_result = sess.run(predict_output, feed_dict=feed_dict)
            opt.run(feed_dict=feed_dict)
            if i % 100 == 0 and i != 0:
                saver.save(sess=sess, save_path='./model/model')
                print('save path finish')


if __name__ == '__main__':
    train()
