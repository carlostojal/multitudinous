import sys
import os
import torch
from torch.utils.data import DataLoader
import argparse
import datetime
import wandb
from typing import Tuple
sys.path.append(".")
from multitudinous.utils.model_builder import build_img_pretraining
from multitudinous.utils.dataset_builder import build_dataset
from multitudinous.utils.loss_builder import build_loss_fn
from multitudinous.configs.pretraining.ImgPreTrainingConfig import ImgPreTrainingConfig
from multitudinous.configs.datasets.DatasetConfig import DatasetConfig
from multitudinous.loss_fns import rmse, rel, delta

def run_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader, device: torch.device, config: ImgPreTrainingConfig = None, mode: str = "train") -> Tuple[float, float]:

    if mode == "train":
        model.train()
    else:
        model.eval()

    criterion = torch.nn.MSELoss(reduction='mean')

    curr_sample = 0
    loss_total = 0.0

    for rgb, depth in loader:

        # zero the gradients for each batch
        optimizer.zero_grad()

        # build the rgbd image
        rgb = rgb.to(device)
        depth = depth.to(device)
        depth = depth.unsqueeze(1)
        rgbd = torch.cat((rgb, depth), dim=1)

        # forward pass
        pred_depth = model(rgbd)

        # compute the loss
        loss = torch.sqrt(criterion(pred_depth, depth))

        # accumulate the loss
        loss_total += loss.item()

        # compute the gradients
        if mode == "train":
            loss.backward()

            # adjust the weights
            optimizer.step()

        curr_sample += config.batch_size

        # log the batch loss
        print(f"\r{mode} sample {curr_sample}/{len(loader)*config.batch_size}, Loss: {loss.item()}", end=" ")

    print()

    # return the last loss and the mean loss
    return loss.item(), loss_total / float(len(loader)*config.batch_size)

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Pre-train the image backbone')
    parser.add_argument('--config', type=str, default='./multitudinous/configs/pretraining/se_resnet50_unet.yaml', help='The image pretraining network to train')
    parser.add_argument('--weights', type=str, default=None, help='The path to the weights of the image backbone')
    parser.add_argument('--dataset', type=str, default='./multitudinous/configs/datasets/tum_rgbd.yaml', help='The dataset to use')
    parser.add_argument('--output', type=str, default='./output', help='The path to save the model weights')
    args = parser.parse_args()

    # build the image pretrainer
    print("Building the image pretrainer...", end=" ")
    config: ImgPreTrainingConfig = ImgPreTrainingConfig()
    config.parse_from_file(args.config)
    img_pretrainer: torch.nn.Module = build_img_pretraining(config.name, 4, args.weights)
    print(img_pretrainer)
    print("done.")

    # initialize wandb
    print("Initializing loggers...", end=" ")
    wandb.init(
        project='img_pretrainer',
        name=f"{config.name}_{datetime.datetime.now().strftime('%H:%M:%S_%Y-%m-%d')}",
        config=config.__dict__
    )
    print("done.")

    # load the dataset
    print("Loading dataset...", end=" ")
    dataset_conf: DatasetConfig = DatasetConfig()
    dataset_conf.parse_from_file(args.dataset)
    train_set, val_set, test_set = build_dataset(dataset_conf)
    print("done.")

    # create the dataloader
    print("Creating the dataloader...", end=" ")
    train_loader: DataLoader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader: DataLoader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True)
    test_loader: DataLoader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)
    print("done.")

    # train the image backbone
    print("Training the image backbone...")

    print("Start time: ", datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"))

    # define the optimizer
    optim = None
    if config.optimizer == 'sgd':
        optim = torch.optim.SGD(img_pretrainer.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer == 'adam':
        optim = torch.optim.Adam(img_pretrainer.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f'Optimizer {config.optimizer} not recognized. Available optimizers are \'sgd\' and \'adam\'.')
    
    # define the loss function
    loss_fn = build_loss_fn(config.loss_fn)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_pretrainer.to(device)

    train_len = len(train_loader)
    val_len = len(val_loader)
    test_len = len(test_loader)

    # train the model
    for epoch in range(config.epochs):

        print(f"Epoch {epoch+1}/{config.epochs}")

        # run one training epoch
        train_loss, train_loss_mean = run_one_epoch(img_pretrainer, optim, train_loader, device, config, "train")

        # log training losses
        wandb.log({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_loss_mean': train_loss_mean
        })

        # run one validation epoch
        val_loss, val_loss_mean = run_one_epoch(img_pretrainer, optim, val_loader, device, config, "val")

        # log validation losses
        wandb.log({
            'epoch': epoch+1,
            'val_loss': val_loss,
            'val_loss_mean': val_loss_mean
        })

        # save the model
        print("Saving the model...", end=" ")
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        pretrainer_path = os.path.join(args.output, f"img_pretrainer_{epoch+1}.pth")
        torch.save(img_pretrainer.state_dict(), pretrainer_path)
        backbone_path = os.path.join(args.output, f"img_backbone_{epoch+1}.pth")
        torch.save(img_pretrainer.encoder.state_dict(), backbone_path)
        print("done.")


    print("Testing the model...")
    
    # test the model
    test_loss, test_loss_mean = run_one_epoch(img_pretrainer, optim, test_loader, device, config, "test")

    # log test losses
    wandb.log({
        'epoch': epoch+1,
        'test_loss': test_loss,
        'test_loss_mean': test_loss_mean
    })

    print("End time: ", datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"))

    # finish logging
    wandb.finish()

    print("done.")
