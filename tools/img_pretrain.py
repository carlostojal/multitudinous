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
import datetime

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
    config: PreTrainingConfig = PreTrainingConfig()
    config.parse_from_file(args.config)
    img_pretrainer: torch.nn.Module = build_img_pretraining(config.name, 4, args.weights)
    print(img_pretrainer)
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

    total_samples = len(dataloader)
    # sample number from which to start validation
    train_thres = int(total_samples * config.train_percent)

    # train the model
    for epoch in range(config.epochs):

        # set the model to training mode
        img_pretrainer.train(True)

        curr_sample = 0

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

            if curr_sample < train_thres:
                # compute the gradients
                loss.backward()

                # adjust the weights
                optim.step()
            else:
                img_pretrainer.eval() # set the model to evaluation mode

            curr_sample += 1

            print(f"\rEpoch {epoch+1}/{config.epochs}, Sample {curr_sample}/{total_samples}, Loss: {loss.item()}", end="")

            del rgb, depth, rgbd, pred_depth, loss_total

            # cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f'\rEpoch {epoch+1}/{config.epochs}, Loss: {loss.item()}')

        # save the model
        print("Saving the model...", end=" ")
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        pretrainer_path = os.path.join(args.output, f"img_pretrainer_{epoch+1}.pth")
        torch.save(img_pretrainer.state_dict(), pretrainer_path)
        backbone_path = os.path.join(args.output, f"img_backbone_{epoch+1}.pth")
        torch.save(img_pretrainer.encoder.state_dict(), backbone_path)
        print("done.")

    print("End time: ", datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"))

    print("done.")

