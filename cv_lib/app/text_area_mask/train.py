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
from app.text_area_mask.bulid_model import MaskModel
import tensorflow as tf

from app.text_area_mask.generate_data import load_batch_training_data


def get_loss(mask_model):
    output = mask_model.output
    output = tf.sigmoid(output)
    label = tf.placeholder(dtype=tf.float32, shape=(None, None, None))
    resize_output = tf.image.resize_bilinear(output, size=[tf.shape(label)[1], tf.shape(label)[2]])
    output = tf.squeeze(resize_output, axis=3)
    loss = tf.reduce_sum(tf.square(output - label))
    return label, loss


def train():
    mask_model = MaskModel()
    mask_model.build()
    input_images = mask_model.input_images
    label, loss = get_loss(mask_model)
    opt = tf.train.AdamOptimizer(learning_rate=0.1 ** 10).minimize(loss)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        training_data = load_batch_training_data(1)
        for i in range(999999999999999):
            print(str(i))
            feed_dict = {input_images: training_data[0], label: training_data[1]}
            sess.run([opt], feed_dict=feed_dict)
            if i % 3 == 0:
                loss_result, opt_result = sess.run([loss, opt], feed_dict=feed_dict)
                print('current step : ' + str(i) + ',loss : ' + str(loss_result))


if __name__ == '__main__':
    train()
