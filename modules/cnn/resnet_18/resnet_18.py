# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     resnet_18
   Description :
   Author :       'li'
   date：          2018/9/6
-------------------------------------------------
   Change Activity:
                   2018/9/6:
-------------------------------------------------
"""
from modules.cnn.resnet_18.config import model_config
from modules.util import cnn_utils
import tensorflow as tf

"""
get a resnet-18 
"""


class ResNet_18(object):
    def __init__(self, batch_size=None, input_images=None,
                 global_step=None, config=model_config):
        """
        initial super parameter
        :param batch_size:
        :param input_images:
        :param global_step:
        :return:
        """
        if input_images is None:
            input_images = tf.placeholder(dtype=tf.float32, shape=(None, 32, 128, 3))
        assert len(input_images.get_shape()) == 4  # shape
        self._batch_size = batch_size
        self._input = input_images
        self._global_step = global_step
        self.kernels = config['kernels']
        self.filters = config['filters']
        self.strides = config['strides']
        self._counted_scope = []
        self._flops = 0
        self._weights = 0
        self.lr = tf.placeholder(tf.float32, name="lr")
        self.is_train = tf.placeholder(tf.bool, name="is_train")

    def build(self):
        """
        build model
        :return:
        """
        print('Building unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(self._input, self.kernels[0], self.filters[0], self.strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block(x, name='conv2_1')
        x = self._residual_block(x, name='conv2_2')
        # conv3_x
        x = self._residual_block_first(x, self.filters[2], self.strides[2], name='conv3_1')
        x = self._residual_block(x, name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, self.filters[3], self.strides[3], name='conv4_1')
        x = self._residual_block(x, name='conv4_2')

        # conv5_x
        x = self._residual_block_first(x, self.filters[4], self.strides[4], name='conv5_1')
        x = self._residual_block(x, name='conv5_2')
        print(x)

    def _relu(self, x, name="relu"):
        x = cnn_utils.relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        """
        conv layers
        :param x:
        :param filter_size:
        :param out_channel:
        :param stride:
        :param pad:
        :param input_q:
        :param output_q:
        :param name:
        :return:
        """
        b, h, w, in_channel = x.get_shape().as_list()
        x = cnn_utils.conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h / stride) * (w / stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        """
        batch normalization
        :param x:
        :param name:
        :return:
        """
        x = cnn_utils.bn(x, self.is_train, self._global_step, name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)


if __name__ == '__main__':
    res = ResNet_18()
    output = res.build()
