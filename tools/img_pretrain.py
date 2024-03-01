import sys
sys.path.append(".")
from multitudinous.utils.model_builder import build_img_backbone
from multitudinous.utils.pretrainer import PreTrainer
from multitudinous.datasets import datasets
import argparse

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Pre-train the image backbone')
    parser.add_argument('--model', type=str, default='se_resnet50', help='The image backbone to train')
    parser.add_argument('--weights', type=str, default=None, help='The path to the weights of the image backbone')
    parser.add_argument('--dataset', type=str, default='cifar10', help='The dataset to use')
    parser.add_argument('--output', type=str, default='output', help='The path to save the model weights')
    args = parser.parse_args()

    # build the image backbone
    img_backbone = build_img_backbone(args.model, args.weights)

    print(img_backbone)

    # TODO: load the dataset
    if args.dataset not in datasets:
        raise ValueError(f"Dataset {args.dataset} not configured")
    dataset = datasets[args.dataset]

    # TODO: create the pretrainer
    # TODO: train the image backbone
    # TODO: save the trained image backbone

