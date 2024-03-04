# MULTITUDINOUS

Voxel mapping from point clouds and RGB-D images using transformers.

## Installation

### Install the torchvision fork

ATTENTION: This is required.


This fork contains extra models developed in the context of MULTITUDINOUS, such as the SE-ResNet-50 and the CBAM-ResNet-50.

- Clone the repository [https://github.com/carlostojal/vision](https://github.com/carlostojal/vision).
- Inside the torchvision repository, run the command `python setup.py install`.

## Pre-training backbones

Pre-training the backbones should make the training of the whole ensemble converge faster, thus being recomended.

### Image Backbone Pre-training
To have a more interactive process, you can use the Jupyter Notebook provided in `notebooks/pretrain_img.ipynb`. To make automated builds or whatever, you can use the script in `tools/img_pretrain.py`. To check the parameters accepted by this script, you can call it with the `--help` argument.

### Point Cloud Backbone Pre-training
Point cloud pre-training is not implemented yet.

## Training

It is recommended to first pre-train the backbones for the training to converge faster.

TODO
