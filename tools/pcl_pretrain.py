import sys
import os
sys.path.append(".")
from multitudinous.utils.model_builder import build_point_cloud_pretraining
from multitudinous.utils.dataset_builder import build_dataset
from multitudinous.utils.loss_builder import build_loss_fn
from multitudinous.configs.pretraining.PclPreTrainingConfig import PclPreTrainingConfig
from multitudinous.configs.datasets.DatasetConfig import DatasetConfig
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
import wandb

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Pre-train the point cloud backbone')
    parser.add_argument('--config', type=str, default='./multitudinous/configs/pretraining/pointnet_seg.yaml', help='The point cloud pretraining network to train')
    parser.add_argument('--weights', type=str, default=None, help='The path to the weights of the point cloud backbone')
    parser.add_argument('--dataset', type=str, default='./multitudinous/configs/datasets/carla.yaml', help='The dataset to use')
    parser.add_argument('--output', type=str, default='./output', help='The path to save the model weights to')
    args = parser.parse_args()

    # build the image pretrainer
    print("Building the point cloud pretrainer...", end=" ")
    config: PclPreTrainingConfig = PclPreTrainingConfig()
    config.parse_from_file(args.config)
    pcl_pretrainer: torch.nn.Module = build_point_cloud_pretraining(config.name, point_dim=config.encoder.point_dim, num_classes=config.num_classes, weights_path=args.weights)
    print(pcl_pretrainer)
    print("done.")

    # initialize wandb
    print("Initializing loggers...", end=" ")
    wandb.init(
        project='pcl_pretrainer',
        name=f"{config.name}_{datetime.datetime.now().strftime('%H:%M:%S_%Y-%m-%d')}",
        config=config.__dict__
    )
    writer = SummaryWriter(f"runs/pcl_pretrainer/{config.name}_{datetime.datetime.now().strftime('%H:%M:%S_%Y-%m-%d')}")
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
        optim = torch.optim.SGD(pcl_pretrainer.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optimizer == 'adam':
        optim = torch.optim.Adam(pcl_pretrainer.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f'Optimizer {config.optimizer} not recognized. Available optimizers are \'sgd\' and \'adam\'.')
    
    # define the loss function
    loss_fn = build_loss_fn(config.loss_fn)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcl_pretrainer.to(device)

    train_len = len(train_loader)
    val_len = len(val_loader)
    test_len = len(test_loader)


    # ------------------- DEBUG -------------------

    """
    rand_pcl = torch.rand((config.batch_size, config.encoder.num_points, config.encoder.point_dim))
    rand_pcl = rand_pcl.to(device)

    print("Random PCL shape: ", rand_pcl.shape)

    pred_seg = pcl_pretrainer(rand_pcl)
    print(pred_seg)
    print(pred_seg.shape)

    # ------------------- DEBUG -------------------
    """

    # train the model
    for epoch in range(config.epochs):

        # set the model to training mode
        pcl_pretrainer.train(True)

        curr_sample = 0

        # iterate samples
        for pcl, seg in train_loader:

            # zero the gradients for each batch
            optim.zero_grad()

            # copy the pointcloud and segmentation to device
            pcl = pcl.to(device)
            seg = seg.to(device)

            # forward pass
            pred_seg = pcl_pretrainer(pcl)

            train_loss_total = 0

            # compute the loss
            for i in range(pred_seg.shape[0]): # batch size
                train_loss_total += loss_fn(pred_seg[i], seg[i])
                curr_sample += 1

            train_loss = train_loss_total / pred_seg.shape[0]
            
            del pcl, seg, pred_seg

            # compute the gradients
            train_loss.backward()

            # adjust the weights
            optim.step()

            print(f"\rEpoch {epoch+1}/{config.epochs}, Sample {curr_sample}/{train_len*config.batch_size}, Train Loss: {train_loss.item()}", end=" ")

        print()

        wandb.log({
            'epoch': epoch+1,
            'train_loss': train_loss.item()
        })
        writer.add_scalar('Loss/train', train_loss.item(), epoch)

        del train_loss_total
        del train_loss
        
        # set the model to evaluation mode
        pcl_pretrainer.eval()

        # evaluate the model

        curr_sample = 0
        for pcl, seg in val_loader:

            # copy the pointcloud and segmentation to device
            pcl = pcl.to(device)
            seg = seg.to(device)

            # forward pass
            pred_seg = pcl_pretrainer(pcl)

            val_loss_total = 0

            # compute the loss
            for i in range(pred_seg.shape[0]): # batch size
                val_loss_total += loss_fn(pred_seg[i], seg[i])

                curr_sample += 1
            
            val_loss = val_loss_total / pred_seg.shape[0]

            del pcl, seg, pred_seg

            print(f"\rEpoch {epoch+1}/{config.epochs}, Sample {curr_sample}/{val_len*config.batch_size}, Val Loss: {val_loss.item()}", end=" ")

        print()

        wandb.log({
            'epoch': epoch+1,
            'val_loss': val_loss.item()
        })
        writer.add_scalar('Loss/val', val_loss.item(), epoch)

        del val_loss_total
        del val_loss

        # save the model
        print("Saving the model...", end=" ")
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        pretrainer_path = os.path.join(args.output, f"pcl_pretrainer_{epoch+1}.pth")
        torch.save(pcl_pretrainer.state_dict(), pretrainer_path)
        backbone_path = os.path.join(args.output, f"pclbackbone_{epoch+1}.pth")
        torch.save(pcl_pretrainer.feature_extractor.state_dict(), backbone_path)
        print("done.")


    curr_sample = 0

    # test the model
    print("Testing the model...", end=" ")
    for pcl, seg in test_loader:

        # copy the pointcloud and segmentation to device
        pcl = pcl.to(device)
        seg = seg.to(device)

        # forward pass
        pred_seg = pcl_pretrainer(pcl)

        test_loss_total = 0
        
        # compute the loss
        for i in range(seg.shape[0]): # batch size
            test_loss_total += loss_fn(pred_seg[i], seg[i])

            curr_sample += 1

        test_loss = test_loss_total / pred_seg.shape[0]

        print(f"\rTesting sample {curr_sample}/{test_len*config.batch_size}, Test Loss: {test_loss.item()}", end=" ")

    print()

    wandb.log({
        'epoch': epoch+1,
        'test_loss': test_loss.item(),
    })
    writer.add_scalar('Loss/test', test_loss.item(), epoch)

    del test_loss_total
    del test_loss

    print("End time: ", datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"))

    # finish logging
    wandb.finish()
    writer.flush()
    writer.close()

    print("done.")

