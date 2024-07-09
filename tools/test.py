import torch
import sys
sys.path.append(".")
from multitudinous.utils.model_builder import build_multitudinous
from multitudinous.configs.model.ModelConfig import ImgBackboneConfig, PointCloudBackboneConfig, NeckConfig, HeadConfig

if __name__ == '__main__':
    # build the multitudinous model
    model = build_multitudinous(img_backbone=ImgBackboneConfig(name='se_resnet50', in_channels=4, img_height=704, img_width=1280),
                                point_cloud_backbone=PointCloudBackboneConfig(name='ndtnet', point_dim=3, num_points=4160, feature_dim=1024),
                                neck=NeckConfig(name='vilbert'),
                                head=HeadConfig(name='transformer', grid_x=500, grid_y=500, grid_z=350))
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # generate random point cloud and rgbd tensors
    point_cloud = torch.rand(1, 4160, 3).to(device)
    rgbd = torch.rand(1, 4, 704, 1280).to(device)

    # forward pass
    output = model(point_cloud, rgbd)
