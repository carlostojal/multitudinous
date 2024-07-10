import torch
from torch import nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import sys
import datetime
from typing import Tuple
import wandb
sys.path.append(".")

from multitudinous.utils.model_builder import build_multitudinous
from multitudinous.configs.model.ModelConfig import ModelConfig
from multitudinous.datasets.CARLA import CARLA
from multitudinous.configs.datasets.DatasetConfig import SubSet, DatasetConfig

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
    for i, (rgbd, pcl, grid) in enumerate(loader):

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
    parser.add_argument('--config', type=str, default='multitudinous/configs/model/se_resnet50-ndtnet.yaml', help='Path to the model YAML configuration file.')
    parser.add_argument('--img_backbone_weights', type=str, default=None, help='Path to the weights of the image backbone')
    parser.add_argument('--point_cloud_backbone_weights', type=str, default=None, help='Path to the weights of the point cloud backbone')
    parser.add_argument('--dataset', type=str, default='multitudinous/configs/datasets/carla.yaml')
    parser.add_argument('--output', type=str, default='output', help='Path to save the model')
    parser.add_argument('--save_every', type=int, default=1, help='Save the model every n epochs')
    args = parser.parse_args()

    # parse the config file
    print("Parsing model configurations...", end=" ")
    config = ModelConfig()
    config.parse_from_file(args.config)
    print("done.")
    print(config)

    # parse the dataset configuration file
    print("Parsing dataset configurations...", end=" ")
    ds_config: DatasetConfig = DatasetConfig()
    ds_config.parse_from_file(args.dataset)
    print("done.")
    print(ds_config)

    # build the model
    print("Building the model...", end=" ")
    model = build_multitudinous(img_backbone_conf=config.img_backbone, 
                                point_cloud_backbone_conf=config.point_cloud_backbone, 
                                neck_conf=config.neck, 
                                head_conf=config.head, 
                                img_backbone_weights_path=args.img_backbone_weights, 
                                point_cloud_backbone_weights_path=args.point_cloud_backbone_weights, 
                                embedding_dim=config.embedding_dim)
    print("done.")

    # initialize wandb
    """
    print("Initializing wandb...", end=" ")
    wandb.init(project="multitudinous", 
               name=f"multitudinous_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
               config=config)
    print("done.")
    """

    # transfer the model to the cpu
    model.to(device)
    # print(model)

    n_img_params = sum(p.numel() for p in model.img_backbone.parameters() if p.requires_grad)
    n_pcl_params = sum(p.numel() for p in model.point_cloud_backbone.parameters() if p.requires_grad)
    n_neck_params = sum(p.numel() for p in model.neck.parameters() if p.requires_grad)
    n_head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Image backbone params: {n_img_params}")
    print(f"Point cloud backbone params: {n_pcl_params}")
    print(f"Neck params: {n_neck_params}")
    print(f"Head params: {n_head_params}")
    print(f"Total params: {n_total_params}")

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # load the datasets
    print("Loading the datasets...", end=" ")
    train_set = CARLA(ds_config, subset=SubSet.TRAIN)
    val_set = CARLA(ds_config, subset=SubSet.VAL)
    test_set = CARLA(ds_config, subset=SubSet.TEST)
    print("done.")

    print("Creating the data loaders...", end=" ")
    generator = torch.Generator(device=device)
    train_loader = DataLoader(train_set, batch_size=int(config.batch_size), shuffle=True, num_workers=1, pin_memory=True, generator=generator, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=int(config.batch_size), shuffle=False, num_workers=1, pin_memory=True, generator=generator, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=(config.batch_size), shuffle=False, num_workers=1, pin_memory=True, generator=generator, persistent_workers=True)
    print("done.")

    for epoch in range(config.epochs):

        print(f"*** EPOCH {epoch+1} ***")

        # train the model
        train_loss, train_loss_mean, train_acc, train_acc_mean = run_one_epoch(model, optimizer, train_loader, device, epoch, mode="train")
        #wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_loss_mean": train_loss_mean, "train_acc": train_acc, "train_acc_mean": train_acc_mean})
        print()

        # validation
        val_loss, val_loss_mean, val_acc, val_acc_mean = run_one_epoch(model, optimizer, val_loader, device, epoch, mode="validation")
        #wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_loss_mean": val_loss_mean, "val_acc": val_acc, "val_acc_mean": val_acc_mean})
        print()

        # save the model (and the backbones, neck and head individually)
        print("Saving the model...")
        torch.save(model.state_dict(), f"{args.output}/multitudinous_{epoch}.pth")
        torch.save(model.img_backbone.state_dict(), f"{args.output}/img_backbone_{epoch}.pth")
        torch.save(model.point_cloud_backbone.state_dict(), f"{args.output}/point_cloud_backbone_{epoch}.pth")
        torch.save(model.neck.state_dict(), f"{args.output}/neck_{epoch}.pth")
        torch.save(model.head.state_dict(), f"{args.output}/head_{epoch}.pth")
        print("done.")

    # test
    print("*** TESTING ***")
    test_loss, test_loss_mean, test_acc, test_acc_mean = run_one_epoch(model, optimizer, test_loader, device, epoch, mode="test")
    #wandb.log({"epoch": epoch+1, "test_loss": test_loss, "test_loss_mean": test_loss_mean, "test_acc": test_acc, "test_acc_mean": test_acc_mean})
    print()

    # finish
    wandb.finish()

    print("Training complete.")
