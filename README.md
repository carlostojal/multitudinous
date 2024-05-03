# MULTITUDINOUS

Voxel mapping from point clouds and RGB-D images using transformers.

## Installation

## Pre-training backbones

Pre-training the backbones should make the training of the whole ensemble converge faster, thus being recomended.

### Image Backbone Pre-training
To have a more interactive process, you can use the Jupyter Notebook provided in `notebooks/pretrain_img.ipynb`. 

However, it is recommended to use the `tools/img_pretrain.py` script. To check the parameters accepted by this script, you can call it with the `--help` argument. 

A YAML configuration file exists for each of the implemented variants. Its hyper-parameters, such as batch size, learning rate and optimizer can be changed in that same files. You can create more variants by creating new YAML files based on the provided examples.

There are also a variety of YAML configuration files for the dataset configurations, and function in a similar manner. You can also define new datasets by creating new YAML files based on the provided examples.

Example: `python tools/img_pretrain.py --config multitudinous/configs/pretraining/se_resnet_unet.yaml --dataset multitudinous/configs/datasets/carla.yaml --output weights/img_pretrain`.

### Point Cloud Backbone Pre-training
Point cloud pre-training is still under implementation.

## Training

It is heavily recommended to first pre-train the backbones for the training to converge faster.

The whole model training is still under implementation.
