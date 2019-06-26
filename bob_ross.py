#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Command line utility to launch the style transfer on a given pre-trained model
"""


import argparse
import logging
import os
from uuid import uuid4
import sys

import docker
from numba import cuda
from PIL import Image
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if not os.path.isdir('./logs'):
    os.mkdir('logs')
if not os.path.isdir('./tmp'):
    os.mkdir('tmp')
    os.mkdir('tmp/ComputeCache')
    os.mkdir('tmp/images')
tf.enable_eager_execution()
_logger = logging.getLogger('bob_ross_ia')
model_choices = ('VGG16', 'VGG19')
dclient = docker.from_env()
cuda.select_device(0)

bob_ross_parser = argparse.ArgumentParser(description='Transfer the style from an image to another')
bob_ross_parser.add_argument('model', help='The pre-trained model to use', choices=model_choices)
bob_ross_parser.add_argument('source_image', help='The pathname source image to apply the style on')
bob_ross_parser.add_argument('style_image', help='The pathname of the style image to use')
bob_ross_parser.add_argument('target', help='The pathname where to store the new image')
bob_ross_parser.add_argument('-n', '--num_iterations', type=int, default=100,
                             help='The number of iterations to apply the transfer')
bob_ross_parser.add_argument('-q', '--quiet', action='store_true', help="Don't log to STDOUT")
bob_ross_parser.add_argument('--content_weight', type=float, default=1e3, help='The weight for the content loss')
bob_ross_parser.add_argument('--style_weight', type=float, default=1e-2, help='The weight for the style loss')
bob_ross_parser.add_argument('--adam_lr', type=int, default=10, help='The learning rate of the Adam optimizer')


if __name__ == '__main__':
    from palette import models
    args = vars(bob_ross_parser.parse_args())
    if not args['quiet']:
        _logger.addHandler(logging.StreamHandler(sys.stdout))
    model_conf = getattr(models, args['model'])
    logging.basicConfig(filename=f"./logs/log_{args['model']}_{args['num_iterations']}.log", level=logging.INFO)
    _logger.info(f"Transferring style with : {args['model']} = {model_conf}")
    transfer_img = models.style_transfer(model_conf, args['source_image'], args['style_image'],
                                         adam_lr=args['adam_lr'], num_iterations=args['num_iterations'])
    cuda.close()
    tmp_img_name = f'{uuid4()}.png'
    tmp_abspth = os.path.abspath('./tmp')
    Image.fromarray(transfer_img).save(f'{tmp_abspth}/images/{tmp_img_name}', 'PNG', optimize=True)
    _logger.info('Applying wx2 ...')
    w2x_command = f"th waifu2x.lua -i /BobRossIAImages/{tmp_img_name} -o /BobRossIAImages/{tmp_img_name}"
    dclient.containers.run(working_dir='/root/waifu2x', runtime='nvidia', auto_remove=True,
                           volumes={f'{tmp_abspth}/ComputeCache': {'bind': '/root/.nv/ComputeCache', 'mode': 'rw'},
                                    f'{tmp_abspth}/images': {'bind': '/BobRossIAImages', 'mode': 'rw'}},
                           image='hvariant/waifu2x', command=w2x_command)
    _logger.info(f"Saving to {args['target']}")
    os.rename(f'{tmp_abspth}/images/{tmp_img_name}', args['target'])
