import torch
import argparse
import sys
sys.path.append(".")

from multitudinous.utils.model_builder import build_multitudinous
from ..configs.model.ModelConfig import ModelConfig

# Run the training

if __name__ == "__main__":

    # detect cuda availability
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default='multitudinous/model_zoo/configs/se_resnet50-pointnet.yaml', help='Path to the model YAML configuration file.')
    parser.add_argument('--img_backbone_weights', type=str, default=None, help='Path to the weights of the image backbone')
    parser.add_argument('--point_cloud_backbone_weights', type=str, default=None, help='Path to the weights of the point cloud backbone')
    parser.add_argument('--output', type=str, default='output', help='Path to save the model')
    args = parser.parse_args()

    # parse the config file
    config = ModelConfig()
    config.parse_from_file(args.config)

    print(config)

    # build the model
    model = build_multitudinous(config.img_backbone, config.point_cloud_backbone, args.img_backbone_weights, args.point_cloud_backbone_weights)

    # transfer the model to the cpu
    model.to(device)

    print(model)

    # TODO

    # generate random input tensors
    random_pcl = torch.rand((1, 10000)).to(device)
    random_rgbd = torch.rand((1, 4, 224, 224)).to(device)

    # test forward pass with random tensors (with correct dimensions)
    out = model(random_pcl, random_rgbd)
