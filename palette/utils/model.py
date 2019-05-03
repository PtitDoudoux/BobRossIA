#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Utility file for configuring the model for the BobRossIA
"""


from typing import Callable, List, Tuple

from tensorflow.keras.models import Model
import tensorflow as tf


def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    """
    Compute the gram matrix of a tensor
    :param input_tensor: The tensor to which we should compute the gram matrix
    :return: The Gram matrix of the input tensor as a tensor
    """
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def model_factory(pre_trained_model: Callable, content_layers: List[str], style_layers: List[str]) -> Model:
    """
    Creates our model with access to intermediate layers.
    This function will load the given model and access the intermediate layers.
    These layers will then be used to create a new model that will take input image
    and return the outputs from these intermediate layers from the given model.

    :param pre_trained_model: The pre-trained keras model factory function to base our model on
    :param content_layers:
    :param style_layers:
    :return: The created tensorflow.keras model that takes image inputs and outputs the style and
             content intermediate layers.
    """
    # Load our model, trained on imagenet data
    model = pre_trained_model(include_top=False, weights='imagenet')
    model.trainable = False
    # Get output layers corresponding to style and content layers
    style_outputs = [model.get_layer(name).output for name in style_layers]
    content_outputs = [model.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model
    return Model(model.input, model_outputs)


def compute_content_loss(base_content: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """
    Compute the loss for the content between two tensors

    Our content loss definition is actually quite simple,
    we simply take the euclidean distance between the two intermediate representations of those images (tensor).

    :param base_content: The base content tensor (aka the source tensor from the model)
    :param target: The tensor target to attain (aka the result gram matrix)
    :return: The content loss as Tensor
    """
    return tf.reduce_mean(tf.square(base_content - target))


def compute_style_loss(base_style: tf.Tensor, gram_target: tf.Tensor) -> tf.Tensor:
    """
    Compute the loss for the style between two tensors
    Expects two images of dimension h, w, c

    Mathematically, we describe the style loss of the base input image, x, and the style image, a,
    as the distance between the style representation (the gram matrices) of these images.

    We describe the style representation of an image as the correlation between different filter responses
    given by the Gram matrix Gl, where Glij is the inner product between the vectorized feature map i and j in layer l.

    We can see that Glij generated over the feature map for a given image represents the correlation
    between feature maps i and j.

    :param base_style: The base style tensor (aka the source tensor from the model)
    :param gram_target: The gram matrix target to attain (aka the result gram matrix)
    :return: The style loss as a tensor
    """
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))  # / (4. * (channels ** 2) * (width * height) ** 2)


def compute_feature_representations(model: tf.keras.Model, img_loader_func: Callable,
                                    content_path: str, style_path: str, num_style_layers: int)\
        -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """
    Helper function to compute our content and style feature representations.

    This function will simply load and preprocess both the content and style
    images from their path. Then it will feed them through the network to obtain
    the outputs of the intermediate layers.

    :param model: The model that we are using.
    :param img_loader_func: The func use which will load the func
    :param content_path: The path to the content image.
    :param style_path: The path to the style image
    :param num_style_layers: The number of style layer in our model
    :return: The style features and the content features.
    """
    # Load our images in
    content_image = img_loader_func(content_path)
    style_image = img_loader_func(style_path)
    # Batch compute content and style features
    style_outputs = model(tf.convert_to_tensor(style_image, dtype=tf.float32))
    content_outputs = model(tf.convert_to_tensor(content_image, dtype=tf.float32))
    # Get the style and content feature representations from our model
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def compute_loss(model: Model, loss_weights: Tuple[float, float], init_image: tf.Variable,
                 gram_style_features: List[tf.Tensor], content_features: List[tf.Tensor],
                 num_style_layers: int, num_content_layers: int)\
        -> Tuple[int, int, int]:
    """
    This function will compute the total loss, style loss and content loss.

    :param model: The model that will give us access to the intermediate layers
    :param loss_weights: The weights of each contribution of each loss function.
        (style weight, content weight, and total variation weight)
    :param init_image: Our initial base image. This image is what we are updating with
        our optimization process. We apply the gradients wrt the loss we are
        calculating to this image.
    :param gram_style_features: Precomputed gram matrices corresponding to the
        defined style layers of interest.
    :param content_features: Precomputed outputs from defined content layers of interest.
    :param num_style_layers: The number of style layer in our model
    :param num_content_layers: The number of content layer in our model
    :return: The total loss, style loss, content loss, and total variational loss
    """
    style_weight, content_weight = loss_weights
    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    style_score = 0
    content_score = 0
    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * compute_style_loss(comb_style[0], target_style)
    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * compute_content_loss(comb_content[0], target_content)
    style_score *= style_weight
    content_score *= content_weight
    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


def compute_grads(cfg: dict) -> Tuple[List[tf.Tensor], Tuple[int, int, int]]:
    """
    Compute the gradient for a given conf
    :param cfg: The configuration to use
    :return:
    """
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    # Compute gradients with input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss
