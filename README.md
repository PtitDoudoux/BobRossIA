# Bob Ross IA Project
## 5IBD ESGI 2018-2019

This project aim to paint an image in the style of another like the artist would have done it.
Aka transfer the style from an image to another.

This project is based on the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

Also we used the code from :
- [Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398)
- [Neural Style Transfer with Eager Execution](https://colab.research.google.com/github/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb)


## How to use it

### Command line help

    usage: bob_ross.py [-h] [-n NUM_ITERATIONS] [--content_weight CONTENT_WEIGHT]
                       [--style_weight STYLE_WEIGHT]
                       {DenseNet121,VGG16,VGG19,Xception} source_image style_image
                       target
    
    Transfer the style from an image to another
    
    positional arguments:
      {DenseNet121,VGG16,VGG19,Xception}
                            The pre-trained model to use
      source_image          The pathname source image to apply the style on
      style_image           The pathname of the style image to use
      target                The pathname where to store the new image
    
    optional arguments:
      -h, --help            show this help message and exit
      -n NUM_ITERATIONS, --num_iterations NUM_ITERATIONS
                            The number of iterations to apply the transfer
      --content_weight CONTENT_WEIGHT
                            The weight for the content loss
      --style_weight STYLE_WEIGHT
                            The weight for the style loss

### Example

    bob_ross.py my_img.jpg my_style.jpg my_transformatted_img.jpg


## TODO
- Handle errors
- Test the project


## Licence