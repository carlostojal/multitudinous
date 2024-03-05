import sys
import os
sys.path.append(".")
from multitudinous.utils.model_builder import build_img_pretraining
from multitudinous.utils.dataset_builder import build_img_dataset
from multitudinous.utils.loss_builder import build_loss_fn
from multitudinous.configs.pretraining.PreTrainingConfig import PreTrainingConfig
from multitudinous.configs.datasets.DatasetConfig import DatasetConfig
import torch
from torch.utils.data import DataLoader
import argparse

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Pre-train the image backbone')
    parser.add_argument('--config', type=str, default='./multitudinous/configs/pretraining/se_resnet50_unet.yaml', help='The image pretraining network to train')
    parser.add_argument('--weights', type=str, default=None, help='The path to the weights of the image backbone')
    parser.add_argument('--dataset', type=str, default='./multitudinous/configs/datasets/tum_rgbd.yaml', help='The dataset to use')
    parser.add_argument('--output', type=str, default='output', help='The path to save the model weights')
    args = parser.parse_args()

    # build the image pretrainer
    print("Building the image pretrainer...", end=" ")
    config: PreTrainingConfig = PreTrainingConfig()
    config.parse_from_file(args.config)
    img_pretrainer: torch.nn.Module = build_img_pretraining(config.name, args.weights)
    print("done.")

    # load the dataset
    print("Loading dataset...", end=" ")
    dataset_conf: DatasetConfig = DatasetConfig()
    dataset_conf.parse_from_file(args.dataset)
    dataset = build_img_dataset(dataset_conf.name, dataset_conf.path)
    print("done.")

    # create the dataloader
    print("Creating the dataloader...", end=" ")
    dataloader: DataLoader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    print("done.")

    # train the image backbone
    print("Training the image backbone...", end=" ")

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
    
    # set the model to training mode
    img_pretrainer.train(True)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_pretrainer.to(device)

    # train the model
    for epoch in range(config.epochs):
        # iterate samples
        for rgb, depth in dataloader:
            # zero the gradients for each batch
            optim.zero_grad()

            # build the rgb-d image
            rgb = rgb.to(device)
            depth = depth.to(device)
            depth = depth.unsqueeze(1)
            rgbd = torch.cat((rgb, depth), dim=1)

            # forward pass
            pred_depth = img_pretrainer(rgbd)

            # compute the loss
            loss_total = 0
            for i in range(len(pred_depth)):
                loss_total += loss_fn(pred_depth[i], depth[i])
            loss = loss_total / len(pred_depth)

            # compute the gradients
            loss.backward()

            # adjust the weights
            optim.step()

            del rgb, depth, rgbd, pred_depth, loss_total, loss

        print(f'Epoch {epoch+1}/{config.epochs}, Loss: {loss.item()}')

        # save the model
        print("Saving the model...", end=" ")
        pretrainer_path = os.path.join(args.output, f"img_pretrainer_{epoch+1}.pth")
        torch.save(img_pretrainer.state_dict(), pretrainer_path)
        backbone_path = os.path.join(args.output, f"img_backbone_{epoch+1}.pth")
        torch.save(img_pretrainer.resnet.state_dict(), backbone_path)

    print("done.")

