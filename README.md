# MULTITUDINOUS

Voxel occupancy mapping from point clouds and RGB-D images using transformers.

Using a ResNet-based image encoder, NDT-Net point cloud encoder, a ViLBERT-based neck and a deconvolutional head.

## Requirements

These are the requirements you need to install on your system in order to train and evaluate the models.

### Bare metal

- Wandb
- PyTorch
- Open3D
- Matplotlib
- NumPy
- OpenCV
- [NDT-Net](https://github.com/carlostojal/NDT-Netpp)

You can install all dependencies except NDT-Net by running the command ```pip install -r requirements.txt```.

### Docker

- Docker

If you want to go this way, you will need to build the container and then you can follow the same instructions. Just use the container as a remote terminal.

### Further notes

You will need to log in to your wandb account to be able to log the losses and accuracies. Run the command ```wandb login```.

## Pre-training the backbones

### Image Backbone Pre-Training

- Confirm the dataset configuration (namely the path) is according to your expectations in the ```multitudinous/configs/datasets/xxx.yaml``` configuration file. 
    - In case you are willing to create/use your own dataset, feel free to create a new file with the same structure.

- Run the command ```python tools/img_pretrain.py --config multitudinous/configs/pretraining/img/se_resnet50_unet.yaml --dataset multitudinous/configs/datasets/carla_rgbd.yaml --output weights/img_pretrain_5k```
    - The first configuration refers to the model configuration. You can check the others available in that same directory or create a new.
    - In any case of doubt, run the script with the ```--help``` option.


### Point Cloud Backbone Pre-Training

The instructions on pre-training the point cloud backbone are described on its README, available in [here](https://github.com/carlostojal/NDT-Netpp).

## Training

It is heavily recommended to first pre-train the backbones for the training to converge faster.

- Run the command ```python tools/train.py --config multitudinous/configs/model/se_resnet50-ndtnet.yaml --img_backbone_weights /path/to/weights --point_cloud_backbone_weights /path/to/weights```.
