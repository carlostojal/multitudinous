import sys
import os
sys.path.append(".")
from multitudinous.utils.dataset_builder import build_img_dataset
from multitudinous.configs.datasets.DatasetConfig import DatasetConfig
from multitudinous.model_index import img_backbones
import torch
from torch.utils.data import DataLoader
import argparse
import datetime

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Quantize an image backbone')
    parser.add_argument('--model', type=str, default='se_resnet50', help='Name of the image backbone to quantize')
    parser.add_argument('--in_channels', type=int, default=4, help='Number of input channels')
    parser.add_argument('--weights', type=str, default=None, help='The path to the weights of the image backbone')
    parser.add_argument('--dataset', type=str, default='./multitudinous/configs/datasets/tum_rgbd.yaml', help='The dataset to use for calibration')
    parser.add_argument('--output', type=str, default='./output', help='The path to save the quantized model weights')
    parser.add_argument('--calib_samples', type=str, default=50, help='The number of samples to use for calibration')
    parser.add_argument('--backend', type=str, default='x86', help='The quantization backend to use. \'x86\' or \'qnnpack\'')
    args = parser.parse_args()

    # build the image pretrainer
    print("Building the image backbone...", end=" ")
    # create the model
    if args.model not in img_backbones:
        raise ValueError(f'Image backbone {args.model} not found. Available image backbones are {list(img_backbones.keys())}.')
    img_backbone = img_backbones[args.model](in_channels=args.in_channels)
    # load the weights
    if args.weights is not None:
        img_backbone.load_state_dict(torch.load(args.weights))
    img_backbone.to('cpu')
    img_backbone.eval()
    print("done.")

    # set the quantization backend
    if args.backend not in ['x86', 'qnnpack']:
        raise ValueError(f'Quantization backend {args.backend} not recognized. Available backends are \'x86\' and \'qnnpack\'.')
    torch.backends.quantized.engine = args.backend
    img_backbone.qconfig = torch.quantization.get_default_qconfig(args.backend)

    # prepare the model for quantization
    print("Preparing the model for quantization...", end=" ")
    img_backbone_prepared = torch.quantization.prepare(img_backbone, inplace=False)
    print("done.")

    # load the dataset
    print("Loading dataset...", end=" ")
    dataset_conf: DatasetConfig = DatasetConfig()
    dataset_conf.parse_from_file(args.dataset)
    dataset = build_img_dataset(dataset_conf.name, dataset_conf.path)
    print("done.")

    # create the dataloader
    print("Creating the dataloader...", end=" ")
    dataloader: DataLoader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("done.")

    # quantize the image backbone
    print("Quantizing the image backbone...")

    total_samples = len(dataloader)
    curr_sample = 0

    # calibrate the model
    for rgb, depth in dataloader:

        # build the rgb-d image
        depth = depth.unsqueeze(1)
        rgbd = torch.cat((rgb, depth), dim=1)

        # forward pass
        out = img_backbone_prepared(rgbd)

        print(f"\rCalibration Sample {curr_sample}/{args.calib_samples}", end="")

        if curr_sample >= args.calib_samples:
            break
        curr_sample += 1

    # convert the model to a quantized model
    img_backbone_quantized = torch.quantization.convert(img_backbone_prepared, inplace=False)

    # save the quantized model
    print("\nSaving the quantized model...", end=" ")
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    torch.save(img_backbone_quantized.state_dict(), os.path.join(args.output, f'{args.model}_quantized.pth'))

    print("done.")
    print(f"Quantized model saved to {os.path.join(args.output, f'{args.model}_quantized.pth')}")