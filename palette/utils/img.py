#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Utility file for dealing with image for the BobRossIA
"""


from typing import Callable, Union

from tensorflow import Tensor
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image


def prepare_image(path_to_img: str, max_dim: int) -> np.ndarray:
    """
    Load an image and prepare it as numpy array
    :param path_to_img: The path to the image to load
    :param max_dim: The max dimension for an image allowed
    :return: The loaded image as np array
    """
    img = load_img(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
    img = img_to_array(img)
    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def load_and_process_img(pre_process: Callable, path_to_img: str, max_dim: int)\
        -> Union[Tensor, np.ndarray]:
    """
    Load and process an image for a given network
    :param pre_process: The pre-process function of Keras / TF pre trained model to use
    :param path_to_img: The path to the image
    :param max_dim: The max dimension for an image allowed for a given network
    :return: The image optimized for the pre-trained network
             as either `tensorflow.Tensor` or `numpy.ndarray`
    """
    img = prepare_image(path_to_img, max_dim)
    img = pre_process(img)
    return img


def deprocess_img(processed_img: np.ndarray) -> np.ndarray:
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
        raise ValueError("Input to deprocess image must be an image of "
                         "dimension [1, height, width, channel] or [height, width, channel]")
    # Perform the inverse of the pre-processing step
    cp_img[:, :, 0] += 103.939
    cp_img[:, :, 1] += 116.779
    cp_img[:, :, 2] += 123.68
    cp_img = cp_img[:, :, ::-1]
    cp_img = np.clip(cp_img, 0, 255).astype('uint8')
    return cp_img
