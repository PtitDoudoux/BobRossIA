#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Define the style transfer func and initialize the models package
"""


from datetime import timedelta
import logging
from time import time
from typing import Callable, List, Tuple, Type

import numpy as np
import tensorflow as tf

from palette.utils.img import deprocess_img
from palette.utils.model import compute_feature_representations, compute_grads, gram_matrix,  model_factory


_logger = logging.getLogger('bob_ross_ia')


def style_transfer(pre_trained_model: Type[tf.keras.Model], content_path: str, style_path: str, content_layers: List[str],
                   style_layers: List[str], lpi: Callable, content_weight=1e3, style_weight=1e-2, num_iterations=1000)\
        -> Tuple[np.ndarray, float]:
    """
    Style transfer from a style image to a source image with a given pre-trained network
    :param pre_trained_model: The pre-trained model to use as source
    :param content_path: The path to the source image to paint the style
    :param style_path: The path to the image to use the style
    :param content_layers: The list of content layers to use
    :param style_layers: The list of style layers to use
    :param lpi: The function to use to load and process image
    :param content_weight: The weight for the content loss
    :param style_weight: The weight for the style loss
    :param num_iterations: The number of iteration to paint
    :return: The best image associated with his best loss
    """
    _logger.info(f'Content weight : {content_weight} | Style Weight : {style_weight}')
    start = time()
    # We don't need to (or want to) train any layers of our model, so we set their
    # trainable to false.
    model = model_factory(pre_trained_model, content_layers, style_layers)
    for layer in model.layers:
        layer.trainable = False
    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = compute_feature_representations(model, lpi, content_path, style_path, len(style_layers))
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    # Set initial image
    init_image = lpi(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
    # Store our best result
    best_loss, best_img = float('inf'), None
    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
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
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())
        _logger.info(f"Iteration n°{i} | loss : {loss} | style_score : {style_score} | content_score : {content_score}")
    computation_time = str(timedelta(seconds=time() - start))
    _logger.info(f'Time Taken : {computation_time}')
    return best_img, best_loss
