import sys
sys.path.append(".")
import torch
import argparse
from multitudinous.utils.model_builder import build_img_backbone


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Export the image backbone to ONNX')
    parser.add_argument('--config', type=str, default='se_resnet50', help='The name of the image backbone to export')
    parser.add_argument('--weights', type=str, default=None, help='The path to the weights of the image backbone')
    parser.add_argument('--output', type=str, default='./output', help='The path to save the ONNX model')
    parser.add_argument('--img_width', type=int, default=640, help='The width of the input image')
    parser.add_argument('--img_height', type=int, default=480, help='The height of the input image')
    parser.add_argument('--img_channels', type=int, default=4, help='The number of channels of the input image')
    args = parser.parse_args()

    # build the image backbone
    print("Building the image backbone...", end=" ")
    img_backbone: torch.nn.Module = build_img_backbone(img_backbone=args.config, in_channels=4, weights_path=args.weights)
    print(img_backbone)
    print("done.")

    # export the image backbone to ONNX
    print("Exporting the image backbone to ONNX...", end=" ")
    img_backbone.eval()
    img_backbone.to(torch.device('cpu'))
    input_tensor = torch.randn(1, args.img_channels, args.img_height, args.img_width)
    torch.onnx.export(img_backbone, input_tensor, args.output, verbose=True)
    print("done.")
