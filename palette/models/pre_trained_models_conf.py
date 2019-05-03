#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
BobRossIA pre-trained models configuration
Use this github helper to see the models summary (aka the layers name, weight) for a model
https://github.com/IoannisNasios/keras_model_summary
TODO: Conf the models : DenseNet121, Xception
"""


from collections import namedtuple
from functools import partial

from tensorflow.keras.applications import densenet, vgg16, vgg19, xception  # tensorflow.python.keras

from palette.utils.img import load_and_process_img


__all__ = ['DenseNet121', 'VGG16', 'VGG19', 'Xception']


PretrainedModelConf = namedtuple('PretrainedModelConf', ['model', 'content_layers', 'style_layers', 'lpi'])


# https://github.com/IoannisNasios/keras_model_summary/blob/master/nbs/DenseNet201.ipynb
DenseNet121 = PretrainedModelConf(densenet.DenseNet121, ['conv5_block16_2_conv'],
                                  [f'conv{i}_block1_0_bn' for i in range(2, 6)],
                                  partial(load_and_process_img, densenet.preprocess_input, max_dim=512))

# https://github.com/IoannisNasios/keras_model_summary/blob/master/nbs/VGG16.ipynb
VGG16 = PretrainedModelConf(vgg16.VGG16, ['block5_conv3'], [f'block{i}_conv1'for i in range(1, 6)],
                            partial(load_and_process_img, vgg16.preprocess_input, max_dim=512))

# https://github.com/IoannisNasios/keras_model_summary/blob/master/nbs/VGG19.ipynb
VGG19 = PretrainedModelConf(vgg19.VGG19, ['block5_conv4'], [f'block{i}_conv1'for i in range(1, 6)],
                            partial(load_and_process_img, vgg19.preprocess_input, max_dim=512))

# https://github.com/IoannisNasios/keras_model_summary/blob/master/nbs/Xception.ipynb
Xception = PretrainedModelConf(xception.Xception, ['block14_sepconv1'], [f'block{i}_sepconv1' for i in range(2, 14)],
                               partial(load_and_process_img, xception.preprocess_input, max_dim=512))
