#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Utility file for dealing with image for the BobRossIA
"""


from typing import Callable

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image


def process_img(image_path: str, preprocess: Callable, max_dim=512, target_size=None) -> np.ndarray:
    """
    Load and process an image for a given network
    :param image_path: The path to the image to process
    :param preprocess: The pre-process function of Keras / TF pre trained model to use
    :param max_dim: The max dimensions of the image which will be resized to it (keep the ratio)
    :param target_size: The size to which the image should be resized, if filled ignore max_dim
    :return: The process image ready for a given network
    """
    img = load_img(image_path)
    if not target_size:
        long = max(img.size)
        scale = max_dim / long
        target_size = (round(img.size[0]*scale), round(img.size[1]*scale))
    img = img.resize(target_size, Image.ANTIALIAS)
    img_arr = img_to_array(img)
    img_arr = img_arr.astype('float64')
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess(img_arr)
    return img_arr


def unprocess_img(processed_img: Image) -> np.ndarray:
    """
    Clean a processed image into a clean and tangible one
    :param processed_img: The processed image to clean
    :exception ValueError: If the image shape is not 3
    :return: The clean image
    """
    cp_img = processed_img.copy()
    if len(cp_img.shape) == 4:
        cp_img = np.squeeze(cp_img, 0)
    if len(cp_img.shape) != 3:
        raise ValueError('Input image to unprocess must be a shape like : (height, width, channel)')
    # Perform the inverse of the pre-processing step aka 'BGR'->'RGB'
    cp_img[:, :, 0] += 103.939
    cp_img[:, :, 1] += 116.779
    cp_img[:, :, 2] += 123.68
    cp_img = cp_img[..., ::-1]
    cp_img = np.clip(cp_img, 0, 255).astype('uint8')
    return cp_img
