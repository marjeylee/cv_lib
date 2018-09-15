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
    output_shape = output.shape.dims[3].value
    output = mask_model.conv_layer(output, 1, output_shape, 1, 1, name='tmp_output')
    label = tf.placeholder(dtype=tf.float32, shape=(None, None, None))
    resize_output = tf.image.resize_bilinear(output, size=[tf.shape(label)[1], tf.shape(label)[2]])
    output = tf.squeeze(resize_output, axis=3)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output)
    loss = tf.reduce_sum(cross_entropy)
    return label, loss, output


def train():
    mask_model = MaskModel()
    mask_model.build()
    input_images = mask_model.input_images
    label, loss, predict_output = get_loss(mask_model)
    conv1_relu = mask_model.conv1_relu
    opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(999999999999999):
            training_data = load_batch_training_data(1)
            feed_dict = {input_images: training_data[0], label: training_data[1]}
            loss_result = sess.run(loss, feed_dict=feed_dict)
            print('current step : ' + str(i) + ',loss : ' + str(loss_result))
            # label_result = sess.run(label, feed_dict=feed_dict)
            # conv1_relu_result = sess.run(conv1_relu, feed_dict=feed_dict)
            # predict_output_result = sess.run(predict_output, feed_dict=feed_dict)
            sess.run([opt], feed_dict=feed_dict)
            if i % 100 == 0:
                saver.save(sess=sess, save_path='./model/model')
                print('save path finish')


if __name__ == '__main__':
    train()
