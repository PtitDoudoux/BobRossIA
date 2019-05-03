#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Command line utility to launch the style transfer on a given pre-trained model
"""


import argparse
import logging
import sys

from PIL import Image
import tensorflow as tf


tf.enable_eager_execution()
_logger = logging.getLogger('bob_ross_ia')
_logger.addHandler(logging.StreamHandler(sys.stdout))
model_choices = ('DenseNet121', 'VGG16', 'VGG19', 'Xception')

bob_ross_parser = argparse.ArgumentParser(description='Transfer the style from an image to another')
bob_ross_parser.add_argument('model', help='The pre-trained model to use', choices=model_choices)
bob_ross_parser.add_argument('source_image', help='The pathname source image to apply the style on')
bob_ross_parser.add_argument('style_image', help='The pathname of the style image to use')
bob_ross_parser.add_argument('target', help='The pathname where to store the new image')
bob_ross_parser.add_argument('-n', '--num_iterations', type=int, default=1000,
                             help='The number of iterations to apply the transfer')
bob_ross_parser.add_argument('--content_weight', type=float, default=1e3,
                             help='The weight for the content loss')
bob_ross_parser.add_argument('--style_weight', type=float, default=1e-2,
                             help='The weight for the style loss')


if __name__ == '__main__':
    from palette import models
    args = vars(bob_ross_parser.parse_args())
    model_conf = getattr(models, args['model'])
    logging.basicConfig(filename=f"./logs/log_{args['model']}_{args['num_iterations']}.log", level=logging.INFO)
    _logger.info(f"Transferring style with : {args['model']} = {model_conf}")
    transferred_img, total_loss = models.style_transfer(model_conf.model, args['source_image'], args['style_image'],
                                                        model_conf.content_layers, model_conf.style_layers,
                                                        model_conf.lpi, num_iterations=args['num_iterations'])
    Image.fromarray(transferred_img).save(args['target'], 'JPEG')
    _logger.info(f"Saving to {args['target']}.jpg")
