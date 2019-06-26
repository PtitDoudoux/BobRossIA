#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Command line utility to launch the style transfer on a given pre-trained model
"""


import asyncio
from base64 import standard_b64encode
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from os import environ, mkdir, path

from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from quart import Quart, request
from quart_cors import cors
from PIL import Image
import tensorflow as tf

from palette import models


environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if not path.isdir('./logs'):
    mkdir('logs')
tf.enable_eager_execution()
app = Quart(__name__)
app = cors(app)
hyper_config = HyperConfig()
hyper_config.bind = ['0.0.0.0:5555']
loop = asyncio.get_event_loop()
th_executor = ThreadPoolExecutor(max_workers=6)


@app.route('/', ['GET'])
async def root():
    """ Root route for the API """
    return await app.send_static_file('index.html')


@app.route('/style_transfer', ['GET', 'POST'])
async def happy_little_accidents():
    """
    Route which execute the style transfer, receive the data from FormData
    :return: The StyleTransfered image as a Quart Response
    """
    form = await request.form
    files = await request.files
    model = form.get('model', 'VGG16')
    num_iterations = int(form.get('num_iterations', 100))
    content_weight = form.get('content_weight', 1e3)
    style_weight = form.get('style_weight', 1e-2)
    adam_lr = form.get('adam_lr', 10)
    source_image = files['source_image']
    style_image = files['style_image']
    model_conf = getattr(models, model)
    # app.logger.info(f'Transferring style with : {model} = {model_conf}')
    transfer_img_args = (th_executor, models.style_transfer, model_conf, source_image.stream,
                         style_image.stream, adam_lr, content_weight, style_weight, num_iterations)
    transfer_img_res = await asyncio.gather(loop.run_in_executor(*transfer_img_args))
    bimg = BytesIO()
    Image.fromarray(transfer_img_res[0]).save(bimg, 'PNG', optimize=True)
    transfer_img = bimg.getvalue()
    bimg.close()
    return standard_b64encode(transfer_img).decode()


if __name__ == '__main__':
    loop.run_until_complete(serve(app, hyper_config))
