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
import cv2

from app.text_area_mask.bulid_model import MaskModel, resize_image
import tensorflow as tf
import numpy as np
from app.text_area_mask.generate_data import load_batch_training_data, input_images_preprocess

# def get_loss(mask_model):
#     output = mask_model.output
#     output_shape = output.shape.dims[3].value
#     output = mask_model.conv_layer(output, 1, output_shape, 1, 1, name='tmp_output')
#     label = tf.placeholder(dtype=tf.float32, shape=(None, None, None))
#     resize_output = tf.image.resize_bilinear(output, size=[tf.shape(label)[1], tf.shape(label)[2]])
#     output = tf.squeeze(resize_output, axis=3)
#     cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output)
#     loss = tf.reduce_sum(cross_entropy)
#     return label, loss, output
from utility import show_img
from utility.image_utility import save_img


def train():
    mask_model = MaskModel()
    mask_model.build()
    input_images = mask_model.input_images
    output = mask_model.output
    output_shape = output.shape.dims[3].value
    output = mask_model.conv_layer(output, 1, output_shape, 1, 1, name='tmp_output')
    output = tf.squeeze(output, axis=0)
    output = tf.sigmoid(output)
    # out_put = tf.sigmoid(output)
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess=sess, save_path='./model/model')
        image_path = 'C:/Users\lr\Desktop/21\H10420180821041413/RIGHT_6.jpg'
        image = cv2.imread(image_path)
        image = input_images_preprocess(image)
        image = resize_image(image)[0]
        image_shape = image.shape
        image = np.reshape(image, [1, image_shape[0], image_shape[1], image_shape[2]])
        feed_dict = {input_images: image}
        loss_result = sess.run(output, feed_dict=feed_dict)
        loss_result = loss_result * 255
        image_result = loss_result.astype(np.int)
        save_img(image_result, 'result.jpg')
        pass


if __name__ == '__main__':
    train()
