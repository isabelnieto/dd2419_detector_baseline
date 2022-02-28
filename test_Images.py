"""Training script for detector."""
from __future__ import print_function

import argparse
from datetime import datetime
import os

import torch
from torch import nn
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import wandb
import numpy as np

import utils
from detector import Detector

NUM_CATEGORIES = 15


def train(device="cpu"):
    """Train the network.

    Args:
        device: The device to train on."""


    # Init model
    detector = Detector().to(device)

    model_path = "/home/isabeln/dd2419_ws/src/m2/scripts/det_2022-02-23_16-55-53-971780.pt"
    utils.load_model(detector,model_path,device)

    dataset = CocoDetection(
        root="./training",
        annFile="./annotations/training.json",
        transforms=detector.input_transform,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)



    # load test images
    # these will be evaluated in regular intervals
    test_images = []
    show_test_images = False
    directory = "./test_images"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(directory, file_name)
            test_image = Image.open(file_path)
            torch_image, _ = detector.input_transform(test_image, [])
            test_images.append(torch_image)

    if test_images:
        test_images = torch.stack(test_images)
        test_images = test_images.to(device)
        show_test_images = True

    print("Training started...")



    detector.eval()
    with torch.no_grad():
        out = detector(test_images).cpu()
        bbs = detector.decode_output(out, 0.5)

        for i, test_image in enumerate(test_images):
            figure, ax = plt.subplots(1)
            ax.imshow(np.asarray(test_image))
            # add bounding boxes
            utils.add_bounding_boxes(ax, bbs[i],dataset.coco.cats)

            plt.close()
    detector.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--cpu", dest="device", action="store_const", const="cpu")
    device.add_argument("--gpu", dest="device", action="store_const", const="cuda")
    args = parser.parse_args()
    train(args.device)
