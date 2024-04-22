import sys
import os
sys.path.append(".")
from multitudinous.utils.model_builder import build_img_pretraining
from multitudinous.utils.dataset_builder import build_dataset
from multitudinous.utils.loss_builder import build_loss_fn
from multitudinous.configs.pretraining.ImgPreTrainingConfig import ImgPreTrainingConfig
from multitudinous.configs.datasets.DatasetConfig import DatasetConfig
from multitudinous.loss_fns import rmse, rel, delta
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
import wandb

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
    writer = SummaryWriter(f"runs/img_pretraniner/{config.name}_{datetime.datetime.now().strftime('%H:%M:%S_%Y-%m-%d')}")
    print("done.")

    # load the dataset
    print("Loading dataset...", end=" ")
    dataset_conf: DatasetConfig = DatasetConfig()
    dataset_conf.parse_from_file(args.dataset)
    train_set, val_set, test_set = build_dataset(dataset_conf.name, dataset_conf.base_path, dataset_conf.train_path, dataset_conf.val_path, dataset_conf.test_path)
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

        # set the model to training mode
        img_pretrainer.train(True)

        curr_sample = 0

        # iterate samples
        for rgb, depth in train_loader:

            # zero the gradients for each batch
            optim.zero_grad()

            # build the rgb-d image
            rgb = rgb.to(device)
            depth = depth.to(device)
            depth = depth.unsqueeze(1)
            rgbd = torch.cat((rgb, depth), dim=1)

            # forward pass
            pred_depth = img_pretrainer(rgbd)

            train_loss_total = 0
            rmse_total = 0
            rel_total = 0
            delta1_total = 0
            delta2_total = 0
            delta3_total = 0

            # compute the loss
            for i in range(pred_depth.shape[0]): # batch size
                train_loss_total += loss_fn(pred_depth[i], depth[i])
                rmse_total += rmse(pred_depth[i], depth[i])
                rel_total += rel(pred_depth[i], depth[i])
                delta1_total += delta(pred_depth[i], depth[i], 1)
                delta2_total += delta(pred_depth[i], depth[i], 2)
                delta3_total += delta(pred_depth[i], depth[i], 3)

                curr_sample += 1

            train_loss = train_loss_total / pred_depth.shape[0]
            rmse_loss = rmse_total / pred_depth.shape[0]
            rel_loss = rel_total / pred_depth.shape[0]
            delta1_loss = delta1_total / pred_depth.shape[0]
            delta2_loss = delta2_total / pred_depth.shape[0]
            delta3_loss = delta3_total / pred_depth.shape[0]

            del rgb, depth, rgbd, pred_depth

            # compute the gradients
            train_loss.backward()

            # adjust the weights
            optim.step()

            print(f"\rEpoch {epoch+1}/{config.epochs}, Sample {curr_sample}/{train_len}, Train Loss: {train_loss.item()}", end=" ")

        print()

        wandb.log({
            'epoch': epoch+1,
            'train_loss': train_loss.item(),
            'rmse': rmse_loss.item(),
            'rel': rel_loss.item(),
            'delta1': delta1_loss.item(),
            'delta2': delta2_loss.item(),
            'delta3': delta3_loss.item()
        })
        writer.add_scalar('Loss/train', train_loss.item(), epoch)
        writer.add_scalar('RMSE/train', rmse_loss.item(), epoch)
        writer.add_scalar('REL/train', rel_loss.item(), epoch)
        writer.add_scalar('Delta1/train', delta1_loss.item(), epoch)
        writer.add_scalar('Delta2/train', delta2_loss.item(), epoch)
        writer.add_scalar('Delta3/train', delta3_loss.item(), epoch)

        del train_loss_total, rmse_total, rel_total, delta1_total, delta2_total, delta3_total
        del train_loss, rmse_loss, rel_loss, delta1_loss, delta2_loss, delta3_loss
        
        # set the model to evaluation mode
        img_pretrainer.eval()

        # evaluate the model

        curr_sample = 0
        for rgb, depth in val_loader:

            # build the rgb-d image
            rgb = rgb.to(device)
            depth = depth.to(device)
            depth = depth.unsqueeze(1)
            rgbd = torch.cat((rgb, depth), dim=1)

            # forward pass
            pred_depth = img_pretrainer(rgbd)

            val_loss_total = 0
            rmse_total = 0
            rel_total = 0
            delta1_total = 0
            delta2_total = 0
            delta3_total = 0

            # compute the loss
            for i in range(pred_depth.shape[0]): # batch size
                val_loss_total += loss_fn(pred_depth[i], depth[i])
                rmse_total += rmse(pred_depth[i], depth[i])
                rel_total += rel(pred_depth[i], depth[i])
                delta1_total += delta(pred_depth[i], depth[i], 1)
                delta2_total += delta(pred_depth[i], depth[i], 2)
                delta3_total += delta(pred_depth[i], depth[i], 3)

                curr_sample += 1
            
            val_loss = val_loss_total / pred_depth.shape[0]
            rmse_loss = rmse_total / pred_depth.shape[0]
            rel_loss = rel_total / pred_depth.shape[0]
            delta1_loss = delta1_total / pred_depth.shape[0]
            delta2_loss = delta2_total / pred_depth.shape[0]
            delta3_loss = delta3_total / pred_depth.shape[0]

            del rgb, depth, rgbd, pred_depth

            print(f"\rEpoch {epoch+1}/{config.epochs}, Sample {curr_sample}/{val_len}, Val Loss: {val_loss.item()}", end=" ")

        print()

        wandb.log({
            'epoch': epoch+1,
            'val_loss': val_loss.item(),
            'rmse': rmse_loss.item(),
            'rel': rel_loss.item(),
            'delta1': delta1_loss.item(),
            'delta2': delta2_loss.item(),
            'delta3': delta3_loss.item()
        })
        writer.add_scalar('Loss/val', val_loss.item(), epoch)
        writer.add_scalar('RMSE/val', rmse_loss.item(), epoch)
        writer.add_scalar('REL/val', rel_loss.item(), epoch)
        writer.add_scalar('Delta1/val', delta1_loss.item(), epoch)
        writer.add_scalar('Delta2/val', delta2_loss.item(), epoch)
        writer.add_scalar('Delta3/val', delta3_loss.item(), epoch)

        del val_loss_total, rmse_total, rel_total, delta1_total, delta2_total, delta3_total
        del val_loss, rmse_loss, rel_loss, delta1_loss, delta2_loss, delta3_loss

        # save the model
        print("Saving the model...", end=" ")
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        pretrainer_path = os.path.join(args.output, f"img_pretrainer_{epoch+1}.pth")
        torch.save(img_pretrainer.state_dict(), pretrainer_path)
        backbone_path = os.path.join(args.output, f"img_backbone_{epoch+1}.pth")
        torch.save(img_pretrainer.encoder.state_dict(), backbone_path)
        print("done.")


    curr_sample = 0

    # test the model
    print("Testing the model...", end=" ")
    for rgb, depth in test_loader:

        # build the rgb-d image
        rgb = rgb.to(device)
        depth = depth.to(device)
        depth = depth.unsqueeze(1)
        rgbd = torch.cat((rgb, depth), dim=1)

        # forward pass
        pred_depth = img_pretrainer(rgbd)

        test_loss_total = 0
        rmse_total = 0
        rel_total = 0
        delta1_total = 0
        delta2_total = 0
        delta3_total = 0
        curr_sample = 0

        # compute the loss
        for i in range(pred_depth.shape[0]): # batch size
            test_loss_total += loss_fn(pred_depth[i], depth[i])
            rmse_total += rmse(pred_depth[i], depth[i])
            rel_total += rel(pred_depth[i], depth[i])
            delta1_total += delta(pred_depth[i], depth[i], 1)
            delta2_total += delta(pred_depth[i], depth[i], 2)
            delta3_total += delta(pred_depth[i], depth[i], 3)

            curr_sample += 1

        test_loss = test_loss_total / pred_depth.shape[0]
        rmse_loss = rmse_total / pred_depth.shape[0]
        rel_loss = rel_total / pred_depth.shape[0]
        delta1_loss = delta1_total / pred_depth.shape[0]
        delta2_loss = delta2_total / pred_depth.shape[0]
        delta3_loss = delta3_total / pred_depth.shape[0]

        print(f"\rTesting sample {curr_sample}/{test_len}, Test Loss: {test_loss.item()}", end=" ")

    print()

    wandb.log({
        'epoch': epoch+1,
        'test_loss': test_loss.item(),
        'rmse': rmse_loss.item(),
        'rel': rel_loss.item(),
        'delta1': delta1_loss.item(),
        'delta2': delta2_loss.item(),
        'delta3': delta3_loss.item()
    })
    writer.add_scalar('Loss/test', test_loss.item(), epoch)
    writer.add_scalar('RMSE/test', rmse_loss.item(), epoch)
    writer.add_scalar('REL/test', rel_loss.item(), epoch)
    writer.add_scalar('Delta1/test', delta1_loss.item(), epoch)
    writer.add_scalar('Delta2/test', delta2_loss.item(), epoch)
    writer.add_scalar('Delta3/test', delta3_loss.item(), epoch)

    del test_loss_total, rmse_total, rel_total, delta1_total, delta2_total, delta3_total
    del test_loss, rmse_loss, rel_loss, delta1_loss, delta2_loss, delta3_loss

    print("End time: ", datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"))

    # finish logging
    wandb.finish()
    writer.flush()
    writer.close()

    print("done.")

