# Bob Ross IA Project
## 5IBD ESGI 2018-2019

This project aim to paint an image in the style of another like the artist would have done it.
Aka transfer the style from an image to another.

This project is based on the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

## References :
- [Neural Style Transfer In Keras](https://markojerkic.com/style-transfer-keras/)
- [AI_Artist](https://github.com/llSourcell/AI_Artist)
- [Making AI Art with Style Transfer using Keras](https://medium.com/mlreview/making-ai-art-with-style-transfer-using-keras-8bb5fa44b216)


## How to use it

### Command line help

    usage: bob_ross.py [-h] [-n NUM_ITERATIONS] [--content_weight {0.01,0.025}]
                   [--style_weight {0.5,1.0,2.0}]
                   [--total_variance_weight TOTAL_VARIANCE_WEIGHT]
                   {VGG16,VGG19} source_image style_image target

    Transfer the style from an image to another
    
    positional arguments:
      {VGG16,VGG19}         The pre-trained model to use
      source_image          The pathname source image to apply the style on
      style_image           The pathname of the style image to use
      target                The pathname where to store the new image
    
    optional arguments:
      -h, --help            show this help message and exit
      -n NUM_ITERATIONS, --num_iterations NUM_ITERATIONS
                            The number of iterations to apply the transfer
      --content_weight {0.01,0.025}
                            The weight for the content loss
      --style_weight {0.5,1.0,2.0}
                            The weight for the style loss
      --total_variance_weight TOTAL_VARIANCE_WEIGHT
                            The weight for the total variance loss


### Example

    python bob_ross.py VGG16 great_sea_turtle.png the_great_wave_off_kanagawa.png great_sea_turtle_kanagawa.png


## TODO
- Handle errors
- Tests
- Propose an API


## Licence
