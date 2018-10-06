# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     bulid_model
   Description :
   Author :       'li'
   date：          2018/9/14
-------------------------------------------------
   Change Activity:
                   2018/9/14:
-------------------------------------------------
"""
import math

import cv2

import tensorflow as tf


def leak_relu(x, leakness=0.2, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x * leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')


class MaskModel(object):

    def __init__(self, is_training=True, mean_pixel_value=None, trainable=True):
        """
        init configuration
        :param is_training:
        :param mean_pixel_value:
        """
        self.trainable = trainable  # variables or constant
        self.is_training = is_training  # use for  batch_normalization
        self.input_images = None
        if mean_pixel_value is not None:
            self.mean_pixel_value = mean_pixel_value
        self.mean_pixel_value = [84.23065114, 102.25350343, 107.33703608]
        self.conv_layer_config = {'kernel_size': 7, 'in_channels': 3, 'out_channels': 64, 'stride': 2}
        self.var_dict = {}
        self.data_dict = None
        self.output = None

    def build(self):
        """
        build model
        :return:
        """
        self.input_images = tf.placeholder(dtype=tf.float32, shape=(None, 512, 512, 3), name='input_images')
        '''2 convolution layers,shorten to a half'''
        # conv_layer = self.conv_layer(self.input_images, 3, 3, 64, 1, "conv1")
        # conv_norm_1 = self.batch_norm(conv_layer)
        # self.conv1_relu = tf.nn.relu(conv_norm_1)
        conv_layer = self.conv_layer(self.input_images, 7, 3, 128, 2, "conv2")
        conv_norm_2 = self.batch_norm(conv_layer)
        self.conv1_relu_2 = tf.nn.relu(conv_norm_2)

        '''nesnet block 1'''
        block1_1 = self.res_block_3_layers(self.conv1_relu_2, [64, 64, 256], "block1_1", True, block_stride=2)
        block1_2 = self.res_block_3_layers(block1_1, [64, 64, 256], "block1_2")
        self.block1_3 = self.res_block_3_layers(block1_2, [64, 64, 256], "block1_3")  # result
        '''merge 1'''
        h1 = self.unpool(self.block1_3)
        # h1 = tf.concat((h1, self.block3_6), axis=-1)
        in_chanel = h1.shape.dims[3].value
        h1 = self.batch_norm(h1)
        h1 = self.conv_layer(h1, 1, in_chanel, 128, 1, "merge1")
        h1 = self.batch_norm(h1)
        h1 = self.conv_layer(h1, 3, 128, 128, 1, name='merge1_1')
        '''result '''
        h1 = tf.concat((h1, self.conv1_relu_2), axis=-1)
        in_chanel = h1.shape.dims[3].value
        h4 = self.conv_layer(h1, 3, in_chanel, 32, 1, "merge4")
        h4 = self.batch_norm(h4)
        output = self.conv_layer(h4, 1, 32, 16, 1, "merge4_1")
        output = self.batch_norm(output)
        self.output = output

    @staticmethod
    def unpool(inputs):
        return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])

    def res_block_3_layers(self, bottom, channel_list, name, change_dimension=False, block_stride=1):
        """
        bottom: input values (X)
        channel_list : number of channel in 3 layers
        name: block name
        """
        if change_dimension:
            block_conv_input = self.conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[2],
                                               block_stride,
                                               name + "_ShortcutConv")
            block_conv_input = self.batch_norm(block_conv_input)
        else:
            block_conv_input = bottom

        block_conv_1 = self.conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[0], block_stride,
                                       name + "_lovalConv1")
        block_norm_1 = self.batch_norm(block_conv_1)
        block_relu_1 = tf.nn.relu(block_norm_1)

        block_conv_2 = self.conv_layer(block_relu_1, 3, channel_list[0], channel_list[1], 1, name + "_lovalConv2")
        block_norm_2 = self.batch_norm(block_conv_2)
        block_relu_2 = tf.nn.relu(block_norm_2)

        block_conv_3 = self.conv_layer(block_relu_2, 1, channel_list[1], channel_list[2], 1, name + "_lovalConv3")
        block_norm_3 = self.batch_norm(block_conv_3)
        block_res = tf.add(block_conv_input, block_norm_3)
        relu = tf.nn.relu(block_res)
        return relu

    def batch_norm(self, inputsTensor):
        """
        Batchnorm
        """
        _BATCH_NORM_DECAY = 0.99
        _BATCH_NORM_EPSILON = 1e-3
        return tf.layers.batch_normalization(inputs=inputsTensor, axis=3, momentum=_BATCH_NORM_DECAY,
                                             epsilon=_BATCH_NORM_EPSILON, center=True, scale=True,
                                             training=self.is_training)

    def max_pool(self, bottom, kernal_size=2, stride=2, name="max"):
        """
        bottom: input values (X)
        kernal_size : n * n kernal
        stride : stride
        name : block_layer name
        """
        print(name + ":")
        print(bottom.get_shape().as_list())
        return tf.nn.max_pool(bottom, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name=name)

    def get_var(self, initial_value, name, idx, var_name):
        """
        load variables from Loaded model or new generated random variables
        initial_value : random initialized value
        name: block_layer name
        index: 0,1 weight or bias
        var_name: name + "_filter"/"_bias"
        """
        if (name, idx) in self.var_dict:
            print("Reuse Parameters...")
            print(self.var_dict[(name, idx)])
            return self.var_dict[(name, idx)]

        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            # var = tf.constant(value, dtype=tf.float32, name=var_name)
            var = tf.Variable(value, name=var_name)
        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        """
        filter_size : 3 * 3
        in_channels : number of input filters
        out_channels : number of output filters
        name : block_layer name
        """
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0,
                                            stddev=1 / math.sqrt(float(filter_size * filter_size)))
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], 0.0, 1.0)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def conv_layer(self, bottom, kernal_size, in_channels, out_channels, stride, name):
        """
        bottom: input values (X)
        kernal_size : n * n kernal
        in_channels: number of input filters
        out_channels : number of output filters
        stride : stride
        name : block_layer name
        """
        print(name + ":")
        print(bottom.get_shape().as_list())
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(kernal_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)

            tf.summary.histogram('weight', filt)
            tf.summary.histogram('bias', conv_biases)

            return bias


def resize_image(im, max_side_len=2400):
    """
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def main():
    mask_model = MaskModel()
    # dir_path = 'F:\dataset\detection_result/results/00f3d476-b62a-11e8-a9c6-11533fbcc673'
    # paths = get_all_files_under_directory(dir_path)
    # images = []
    # for p in paths:
    #     if p.find('.png') > 0:
    #         images.append(cv2.imread(p))
    #
    # prece = input_images_preprocess(images)
    mask_model.build()


if __name__ == '__main__':
    main()
