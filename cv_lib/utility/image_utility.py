# -*- coding: utf-8 -*-
from utility.file_path_utility import get_all_files_under_directory

__author__ = 'l'
__date__ = '2018/5/24'
import cv2 as cv
import numpy as np


def show_img(new_img):
    """
    显示图片
    :param new_img:
    :return:
    """
    cv.imshow("Image", new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def save_img(img, path):
    """
    保存图片
    :param img:图片
    :param path: 路径
    :return:
    """
    cv.imwrite(path, img)


def get_one_image_average_value(img):
    """
    rgb
    :param img:
    :return:
    """
    average = np.mean(img, axis=0)
    average = np.mean(average, axis=0)
    return average


def get_images_average_value(images_path):
    """
    get images average values
    :param images_path:
    :return:
    """
    images_size = len(images_path)
    print('images size:' + str(images_size))
    total_value = np.zeros((3,), dtype=np.float)
    add_time = 0
    for index in range(images_size):
        if index % 100 == 0:
            print(str(index) + '/' + str(images_size))
        p = images_path[index]
        img = cv.imread(p)
        if img is not None:
            aver = get_one_image_average_value(img)
            total_value = np.add(total_value, aver)
            add_time = add_time + 1
    average_value = total_value / add_time
    return average_value


if __name__ == '__main__':
    dir_path = 'F:\BaiduNetdiskDownload/uuid_image'
    paths = get_all_files_under_directory(dir_path)
    average_value = get_images_average_value(paths)
    print(average_value)
