#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Define the style transfer func and initialize the models package
"""


from datetime import timedelta
import logging
from time import time
from typing import Callable, List

import numpy as np
import tensorflow as tf

from palette.models.pre_trained_models_conf import PretrainedModelConf
from palette.utils.img import deprocess_img
from palette.utils.model import compute_feature_representations, compute_grads, gram_matrix,  model_factory


_logger = logging.getLogger('bob_ross_ia')


def style_transfer(pre_trained_model: PretrainedModelConf, content_path: str, style_path: str, adam_lr=10,
                   content_weight=1e3, style_weight=1e-2, num_iterations=100) -> np.ndarray:
    """
    Style transfer from a style image to a source image with a given pre-trained network
    :param pre_trained_model: The pre-trained model to use as source
    :param content_path: The path to the source image to paint the style
    :param style_path: The path to the image to use the style
    :param adam_lr: The learning rate of the Adam optimizer
    :param content_weight: The weight for the content loss
    :param style_weight: The weight for the style loss
    :param num_iterations: The number of iteration to paint
    :return: The best image associated with his best loss
    """
    _logger.info(f'Content weight : {content_weight} | Style Weight : {style_weight}')
    start = time()
    # We don't need to (or want to) train any layers of our model, so we set their
    model = model_factory(pre_trained_model.model, pre_trained_model.content_layers, pre_trained_model.style_layers)
    for layer in model.layers:
        layer.trainable = False
    # Create our optimizer
    adam_opt = tf.train.AdamOptimizer(learning_rate=adam_lr, beta1=0.9, epsilon=1e-1)
    # Set initial image
    gen_img = tf.Variable(pre_trained_model.lpi(content_path), dtype=tf.float32, name='gen_img')
    _st(model, gen_img, content_path, style_path, pre_trained_model.content_layers,
        pre_trained_model.style_layers, pre_trained_model.lpi, adam_opt, content_weight,
        style_weight, num_iterations)
    transfer_img = deprocess_img(gen_img.numpy())
    computation_time = str(timedelta(seconds=time() - start))
    _logger.info(f'Time Taken : {computation_time}')
    return transfer_img


def _st(model: tf.keras.Model, gen_img: tf.Variable, content_path: str, style_path: str,
        content_layers: List[str], style_layers: List[str], lpi: Callable, opt: tf.train.AdamOptimizer,
        content_weight=1e3, style_weight=1e-2, num_iterations=100) -> None:
    """
    Style transfer from a style image to a source image with a given pre-trained network
    :param model: The model to use for the style transfer
    :param gen_img: The generated image to modify INPLACE
    :param content_path: The path to the source image to paint the style
    :param style_path: The path to the image to use the style
    :param content_layers: The list of content layers to use
    :param style_layers: The list of style layers to use
    :param lpi: The function to use to load and process image
    :param opt: The Adam optimizer to use
    :param content_weight: The weight for the content loss
    :param style_weight: The weight for the style loss
    :param num_iterations: The number of iteration to paint
    :return: The best image associated with his best loss
    """
    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = compute_feature_representations(model, lpi, content_path, style_path, len(style_layers))
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'gen_img': gen_img,
        'gram_style_features': gram_style_features,
        'content_features': content_features,
        'num_style_layers': len(style_layers),
        'num_content_layers': len(content_layers)
    }
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, gen_img)])
        clipped = tf.clip_by_value(gen_img, min_vals, max_vals)
        gen_img.assign(clipped)
        _logger.info(f"Iteration n°{i} | loss : {loss} | style_score : {style_score} | content_score : {content_score}")