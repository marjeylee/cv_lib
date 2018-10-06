# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     generate_data
   Description :
   Author :       'li'
   date：          2018/9/14
-------------------------------------------------
   Change Activity:
                   2018/9/14:
-------------------------------------------------
"""
import cv2
import os

from app import log_util
from app.text_area_mask.bulid_model_backkk import resize_image
from utility import show_img
from utility.file_path_utility import get_all_files_under_directory
from utility.image_utility import save_img

__author__ = 'li'
import numpy as np


def input_images_preprocess(input_images):
    """
    images means images list not a ndarray,every image is a ndarray object
    :param input_images:
    :return:
    """
    mean_pixel_value = [84.23065114, 102.25350343, 107.33703608]
    images_size = len(input_images)
    assert images_size > 0
    mean_images = []
    shape = input_images.shape
    r, g, b = input_images[:, :, 0] - mean_pixel_value[0], \
              input_images[:, :, 1] - mean_pixel_value[1], \
              input_images[:, :, 2] - mean_pixel_value[2]
    r = r.reshape((shape[0], shape[1], 1))
    g = g.reshape((shape[0], shape[1], 1))
    b = b.reshape((shape[0], shape[1], 1))
    img = np.concatenate((r, g, b), axis=2)
    mean_images.append(img)
    return np.concatenate(mean_images)


def load_training_data(random_keys):
    """
    :param random_keys:
    :return:
    """
    images = []
    labels = []
    for k in random_keys:
        image_path = image_mapping[k]
        label_path = label_mapping[k]
        image = cv2.imread(image_path)

        shape = image.shape
        mask = np.zeros((shape[0], shape[1]))
        with open(label_path, mode='r', encoding='utf8') as file:
            lines = file.readlines()
            for line in lines:
                if len(line) > 5:
                    columns = line.split(',')
                    mask[int(columns[1]):int(columns[5]), int(columns[0]):int(columns[4])] = 1
        image = resize_image(image)[0]
        image = cv2.resize(image, (512, 512))
        shape = image.shape
        mask = cv2.resize(mask, (int(shape[1] / 2), int(shape[0] / 2)))
        # save_img(image, 'img.jpg')
        # save_img(mask.astype(np.int) * 255, 'mask.jpg')
        image = input_images_preprocess(image)
        mask_shape = mask.shape
        image = image.reshape([1, shape[0], shape[1], shape[2]])
        mask = mask.reshape((1, mask_shape[0], mask_shape[1]))
        images.append(image)
        labels.append(mask)
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)


def load_mapping(image_dir_path):
    """
    load  name and path map
    :param image_dir_path:
    :return:
    """
    imgs_path = get_all_files_under_directory(image_dir_path)
    mapping = {}
    for p in imgs_path:
        _, name = os.path.split(p)
        name = name.split('.')[0]
        mapping[name] = p
    return mapping


def filter_mapping(image_mapping, label_mapping):
    """
    filter
    :param image_mapping:
    :param label_mapping:
    :return:
    """
    image_keys = image_mapping.keys()
    label_keys = label_mapping.keys()
    image_keys = list(image_keys)
    for k in image_keys:
        if k not in label_keys:
            del image_mapping[k]


image_dir_path = '/gpu_data/code/detection/data/ICPR2018/'
label_dir_path = '/gpu_data/code/detection/data/ICPR2018/'


# image_dir_path = 'E:\dataset\detection/'
# label_dir_path = 'E:\dataset\detection/'


def load_image_mapping(image_dir_path):
    """
        load  name and path map
        :param image_dir_path:
        :return:
        """
    imgs_path = get_all_files_under_directory(image_dir_path)
    mapping = {}
    for p in imgs_path:
        if p.find('.jpg') > 0:
            _, name = os.path.split(p)
            name = name.split('.')[0]
            mapping[name] = p
    return mapping


image_mapping = load_image_mapping(image_dir_path)


def load_txt_mapping(label_dir_path):
    """
        load  name and path map
        :param image_dir_path:
        :return:
        """
    imgs_path = get_all_files_under_directory(label_dir_path)
    mapping = {}
    for p in imgs_path:
        if p.find('.txt') > 0:
            _, name = os.path.split(p)
            name = name.split('.')[0]
            mapping[name] = p
    return mapping


label_mapping = load_txt_mapping(label_dir_path)
filter_mapping(image_mapping, label_mapping)
image_names = list(image_mapping.keys())
data_index = 0

training_data_pool = []


def add_data_to_pool():
    global training_data_pool
    training_data_pool = image_names[:data_index]


def load_batch_training_data(batch_size):
    # global data_index
    # if data_index == 0:
    #     add_training_data(batch_size)
    random_keys = np.random.choice(image_names, size=batch_size)
    # random_keys = np.random.choice(training_data_pool, size=batch_size)
    training_data = load_training_data(random_keys)
    return training_data


# def add_training_data(batch_size):
#     pass
#     global data_index
#     data_index = data_index + batch_size
#     add_data_to_pool()
#     log_util.info('current training data size :' + str(len(training_data_pool)))


def main():
    batch_size = 1
    trainging_data = load_batch_training_data(batch_size)
    pass


if __name__ == '__main__':
    main()
