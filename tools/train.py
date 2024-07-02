import torch
from torch import nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import sys
from typing import Tuple
import wandb
sys.path.append(".")

from multitudinous.utils.model_builder import build_multitudinous
from multitudinous.configs.model.ModelConfig import ModelConfig

def run_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, 
                  loader: DataLoader, device: torch.device,
                  epoch: int, mode: str="train") -> Tuple[float, float, float, float]:
    
    # set the model to train or eval mode
    if mode == "train":
        model.train()
    else:
        model.eval()

    # initialize the loss
    loss_total = 0
    acc_total = 0
    curr_sample = 0
    for i, (pcl, rgbd, grid) in enumerate(loader):

        # transfer the data to the device
        pcl = pcl.to(device)
        rgbd = rgbd.to(device)
        grid = grid.to(device)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        out = model(pcl, rgbd)

        # instantiate the criterion
        criterion = nn.CrossEntropyLoss(reduction='mean')

        # calculate the loss
        loss = criterion(out, grid)

        # backpropagation
        if mode == "train":
            loss.backward()
            optimizer.step()

        # calculate the accuracy. every voxel with more than 0.5 probability is considered occupied
        acc = ((out > 0.5) == grid).sum().item() / grid.numel()
        acc_total += acc

        # accumulate the mean loss per sample
        loss_total += loss.item()

        # increment the current sample
        curr_sample += loader.batch_size

        # print the loss
        print(f"\r{mode} epoch {epoch+1} ({curr_sample}/{len(loader)*loader.batch_size}): loss={loss.item()}, acc={acc}", end="")

    avg_loss = loss_total / (len(loader) * loader.batch_size)
    avg_acc = acc_total / (len(loader) * loader.batch_size)

    return loss.item(), avg_loss, acc, avg_acc


# Run the training

if __name__ == "__main__":

    # detect cuda availability
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # set the detected device as default
    torch.set_default_device(device)

    # parse command line arguments
    parser = ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default='multitudinous/configs/model/se_resnet50-pointnet.yaml', help='Path to the model YAML configuration file.')
    parser.add_argument('--img_backbone_weights', type=str, default=None, help='Path to the weights of the image backbone')
    parser.add_argument('--point_cloud_backbone_weights', type=str, default=None, help='Path to the weights of the point cloud backbone')
    parser.add_argument('--output', type=str, default='output', help='Path to save the model')
    parser.add_argument('--save_every', type=int, default=1, help='Save the model every n epochs')
    args = parser.parse_args()

    # parse the config file
    config = ModelConfig()
    config.parse_from_file(args.config)

    print(config)

    # build the model
    model = build_multitudinous(config.img_backbone, config.point_cloud_backbone, config.neck, config.head, args.img_backbone_weights, args.point_cloud_backbone_weights)

    # initialize wandb
    wandb.init(project="multitudinous", config=config, name="multitudinous")

    # transfer the model to the cpu
    model.to(device)
    print(model)

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # TODO: load the dataset

    for epoch in range(config.epochs):

        # train the model
        train_loss, train_loss_mean, train_acc, train_acc_mean = run_one_epoch(model, optimizer, train_loader, device, epoch, mode="train")
        print()

        # validation
        val_loss, val_loss_mean, val_acc, val_acc_mean = run_one_epoch(model, optimizer, val_loader, device, epoch, mode="validation")
        print()

        # save the model (and the backbones, neck and head individually)
        torch.save(model.state_dict(), f"{args.output}/multitudinous_{epoch}.pth")
        torch.save(model.img_backbone.state_dict(), f"{args.output}/img_backbone_{epoch}.pth")
        torch.save(model.point_cloud_backbone.state_dict(), f"{args.output}/point_cloud_backbone_{epoch}.pth")
        torch.save(model.neck.state_dict(), f"{args.output}/neck_{epoch}.pth")
        torch.save(model.head.state_dict(), f"{args.output}/head_{epoch}.pth")

    # test
    test_loss, test_loss_mean, test_acc, test_acc_mean = run_one_epoch(model, optimizer, test_loader, device, epoch, mode="test")
    print()

    print("Training complete.")
