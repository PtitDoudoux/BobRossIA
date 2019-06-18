#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Define the Usable models to use in the style transfer
"""


import logging
from time import time
from typing import Callable, List, Tuple

from keras import backend as K
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .pre_trained_conf import PretrainedModelConf
from palette.utils.img import process_img, unprocess_img


__all__ = ['StyleTransferModel', 'style_transfer']
_logger = logging.getLogger('bob_ross_ia')


class StyleTransferModel:

    def __init__(self, model_conf: PretrainedModelConf, content_path: str, style_path: str,
                 content_weight=0.01, style_weight=1.0, tv_weight=1.0):
        """
        Initialize the StyleTransferModel class
        :param model_conf: The model conf to use
        :param content_path: The path to the content image
        :param style_path: The path to the style image
        :param content_weight: Weight of content loss
        :param style_weight: Weight of style loss
        :param tv_weight: Weight for the total variance loss
        """
        self.tf_session = K.get_session()
        self.model_conf = model_conf
        self.model = self._build_model(model_conf, content_path, style_path)
        self.outputs_dict = dict(((layer.name, layer.output) for layer in self.model.layers))
        self.layers_weight = np.ones(len(self.model_conf.style_layers)) / float(len(self.model_conf.style_layers))
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self._loss = None

    def _build_model(self, model_conf: PretrainedModelConf, content_path: str, style_path: str) -> 'keras.Model':
        """
        Build a Keras model based on a configuration
        :param model_conf: The model configuration to build on
        :param content_path: The path to the content image
        :param style_path: The path to the style image
        :return: The builded keras model
        """
        arr_content_img = process_img(content_path, model_conf.preprocess)
        content_img = K.variable(arr_content_img)
        self.target_height = arr_content_img.shape[1]
        self.target_width = arr_content_img.shape[2]
        arr_style_image = process_img(style_path, model_conf.preprocess,
                                      target_size=(self.target_width, self.target_height))
        style_img = K.variable(arr_style_image)
        self.gen_img = K.placeholder(shape=(1, self.target_height, self.target_width, 3))
        input_tensor = K.concatenate([content_img, style_img, self.gen_img], axis=0)
        return model_conf.model(include_top=False, weights='imagenet', input_tensor=input_tensor)

    @staticmethod
    def gram_matrix(features_matrix: np.ndarray) -> 'tensorflow.Tensor':
        """
        Compute the gram matrix of a features representation
        The Gram matrix is the dot product of the flattened feature map and the transpose of the flattened feature map.
        :param features_matrix: The features matrix to use
        :return: A Tensor representing the gram matrix
        """
        if K.image_data_format() == 'channels_first':
            features_matrix = K.flatten(features_matrix)
        else:
            features_matrix = K.batch_flatten(K.permute_dimensions(features_matrix, (2, 0, 1)))
        # Dot product of the flattened feature map and the transpose of the flattened feature map
        return K.dot(features_matrix, K.transpose(features_matrix))

    @staticmethod
    def content_loss(features_content_matrix: np.ndarray, features_gen_matrix: np.ndarray) -> 'tensorflow.Tensor':
        """
        Get the content loss for a given content against the applied style

        The purpose of the content loss is to make sure that the generated image x
            retains some of the global characteristics of the content image, p
        The content loss is the simple Euclidean distance between the outputs of the model for the content image
        and the generated image at a specific layer in the network.

        :param features_content_matrix: The feature matrix representing the content image
        :param features_gen_matrix: The feature matrix representing the generated image
        :return: The computed content loss as a Tensor
        """
        return K.sum(K.square(features_gen_matrix - features_content_matrix))

    def style_loss(self, features_style_matrix: np.ndarray, features_gen_matrix: np.ndarray) -> 'tensorflow.Tensor':
        """
        Compute the style loss for the generated image against the style image for all styles layers

        Ascending layers in most convolutional networks such as VGG have increasingly larger receptive fields.
        As this receptive field grows, more large-scale characteristics of the input image are preserved.
        Because of this, multiple layers should be selected for “style”
            to incorporate both local and global stylistic qualities.
        To create a smooth blending between these different layers, we can assign a weight w to each layer

        :param features_style_matrix: The feature matrix representing the style  image
        :param features_gen_matrix: The feature matrix representing the generated image
        :return: The computed style loss as a Tensor
        """
        style_gram_matrix = self.gram_matrix(features_style_matrix)
        gen_gram_matrix = self.gram_matrix(features_gen_matrix)
        channels = 3
        size = self.target_height * self.target_width
        # Euclidean distance of the gram matrices multiplied by the constant
        return K.sum(K.square(style_gram_matrix - gen_gram_matrix)) / (4. * (channels ** 2) * (size ** 2))

    def total_loss(self, features_gen_matrix: np.ndarray) -> 'tensorflow.Tensor':
        """
        Calculate the total variance loss
        This loss reduces the amount of noise in the generated image
        :param features_gen_matrix: The feature matrix representing the generated image
        :return: The total variance loss as a Tensor
        """
        if K.ndim(features_gen_matrix) != 4:
            raise ValueError("The features matrix for the generated image should be 4")
        a = K.square(features_gen_matrix[:, :self.target_height - 1, :self.target_width - 1, :]
                     - features_gen_matrix[:, 1:, :self.target_width - 1, :])
        b = K.square(features_gen_matrix[:, :self.target_height - 1, :self.target_width - 1, :]
                     - features_gen_matrix[:, :self.target_height - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    def calculate_loss_and_grads(self, features_gen_matrix: np.ndarray, func_outputs: Callable)\
            -> Tuple[float, 'tensorflow.Tensor']:
        """
        Compute loss and grads simultaneously
        :param features_gen_matrix: The feature matrix representing the generated image
        :param func_outputs: The function to calculate the outputs
        :return: The computed loss and grad as Tuple
        """
        if K.image_data_format() == 'channels_first':
            features_gen_matrix = features_gen_matrix.reshape((1, 3, self.target_height, self.target_width))
        else:
            features_gen_matrix = features_gen_matrix.reshape((1, self.target_height, self.target_width, 3))
        outs = func_outputs([features_gen_matrix])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    @property
    def loss(self) -> float:
        """
        Compute the loss on the generated image compared to the content and style image
        :return: The total loss
        """
        if self._loss is not None:
            return self._loss
        loss = 0.
        layer_features = self.outputs_dict[self.model_conf.content_layers]
        content_img_features = layer_features[0, :, :, :]
        gen_img_features = layer_features[2, :, :, :]
        loss += self.content_weight * self.content_loss(content_img_features, gen_img_features)
        feature_layer_names = self.model_conf.style_layers
        for name in feature_layer_names:
            layer_features = self.outputs_dict[name]
            style_features = layer_features[1, :, :, :]
            gen_img_features = layer_features[2, :, :, :]
            s1 = self.style_loss(style_features, gen_img_features)
            # We need to divide the loss by the number of layers that we take into account
            loss += (self.style_weight / len(feature_layer_names)) * s1
        loss += self.tv_weight * self.total_loss(self.gen_img)
        self._loss = loss
        return loss

    @property
    def grads(self) -> 'tensorflow.Tensor':
        """
        Compute the grads for the model
        :return: The gradient as a Tensor
        """
        return K.gradients(self.loss, self.gen_img)


class Evaluator:
    """
    This Evaluator class makes it possible to compute loss and gradients in one pass while retrieving them
        via two separate functions, "loss" and "grads".
    This is done because scipy.optimize requires separate functions for loss and gradients,
        but computing them separately would be inefficient.
    """

    def __init__(self, sfm: StyleTransferModel):
        """
        Initialize the Evaluator class
        :param sfm: The StyleTransferModel object to use
        """
        self.loss_value = None
        self.grads_value = None
        self.sfm = sfm
        self.func_outputs = self._func_outputs()

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_value = self.sfm.calculate_loss_and_grads(x, self.func_outputs)
        self.loss_value = loss_value
        self.grads_value = grad_value
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grads_values = np.copy(self.grads_value)
        self.loss_value = None
        self.grads_value = None
        return grads_values

    def _func_outputs(self) -> Callable:
        """
        Determine the func output for the evaluator
        :return: The callable func outputs for the Evaluator
        """
        outputs = [self.sfm.loss]
        if isinstance(self.sfm.grads, (list, tuple)):
            outputs += self.sfm.grads
        else:
            outputs.append(self.sfm.grads)
        return K.function([self.sfm.gen_img], outputs)


def style_transfer(sfm: StyleTransferModel, gen_path: str = None, iteration=250) -> np.ndarray:
    """
    Run the Style transfer algorithm
    :param sfm: The StyleTransferModel to use
    :param gen_path: The path to the content path to use as the base for the generated image
        if None create a random image
    :param iteration: The number of iteration to run the algorithm
    :return: The generated image as a numpy array
    """
    loss = float('inf')
    evaluator = Evaluator(sfm)
    if gen_path:
        gen_img = process_img(gen_path, sfm.model_conf.preprocess)
    else:
        gen_img = np.random.uniform(0, 255, (1, 3, sfm.target_width, sfm.target_height))
    start_time = time()
    for i in range(iteration):
        new_gen_img, new_loss, info = fmin_l_bfgs_b(evaluator.loss, gen_img.flatten(), fprime=evaluator.grads, maxfun=20)
        if new_loss < loss:
            gen_img = new_gen_img
            loss = new_loss
        _logger.info(f'Iteration : {i} | Loss : {loss}')
    _logger.info(f'Done in : {time() - start_time}')
    return unprocess_img(gen_img.reshape((sfm.target_height, sfm.target_width, 3)))
