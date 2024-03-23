# MULTITUDINOUS

Voxel mapping from point clouds and RGB-D images using transformers.

## Installation

## Pre-training backbones

Pre-training the backbones should make the training of the whole ensemble converge faster, thus being recomended.

### Image Backbone Pre-training
To have a more interactive process, you can use the Jupyter Notebook provided in `notebooks/pretrain_img.ipynb`. However, it is recommended to use the `tools/img_pretrain.py` script. To check the parameters accepted by this script, you can call it with the `--help` argument.

### Point Cloud Backbone Pre-training
Point cloud pre-training is not implemented yet.

## Training

It is recommended to first pre-train the backbones for the training to converge faster.

TODO
