#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Command line utility to launch the style transfer on a given pre-trained model
"""


import argparse
import logging
import sys

from PIL import Image

from palette.models import style_transfer, StyleTransferModel


_logger = logging.getLogger('bob_ross_ia')
_logger.addHandler(logging.StreamHandler(sys.stdout))
model_choices = ('VGG16', 'VGG19')

bob_ross_parser = argparse.ArgumentParser(description='Transfer the style from an image to another')
bob_ross_parser.add_argument('model', help='The pre-trained model to use', choices=model_choices)
bob_ross_parser.add_argument('source_image', help='The pathname source image to apply the style on')
bob_ross_parser.add_argument('style_image', help='The pathname of the style image to use')
bob_ross_parser.add_argument('target', help='The pathname where to store the new image')
bob_ross_parser.add_argument('-n', '--num_iterations', type=int, default=250,
                             help='The number of iterations to apply the transfer')
bob_ross_parser.add_argument('--content_weight', type=float, default=0.01, choices=(0.01, 0.025),
                             help='The weight for the content loss')
bob_ross_parser.add_argument('--style_weight', type=float, default=1.0, choices=(0.5, 1.0, 2.0),
                             help='The weight for the style loss')
bob_ross_parser.add_argument('--total_variance_weight', type=float, default=1.0,
                             help='The weight for the total variance loss')


if __name__ == '__main__':
    from palette import models
    args = vars(bob_ross_parser.parse_args())
    model_conf = getattr(models, args['model'])
    logging.basicConfig(filename=f"./logs/log_{args['model']}_{args['num_iterations']}.log", level=logging.INFO)
    _logger.info(f"Transferring style with : {args['model']} = {model_conf}")
    sfm = StyleTransferModel(model_conf, args['source_image'], args['style_image'], args['content_weight'],
                             args['style_weight'], args['total_variance_weight'])
    transferred_img = style_transfer(sfm, args['source_image'], args['num_iterations'])
    Image.fromarray(transferred_img).save(args['target'], 'PNG', optimize=True)
    _logger.info(f"Saving to {args['target']}")
