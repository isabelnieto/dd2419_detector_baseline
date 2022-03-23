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

from pycocotools.cocoeval import COCOeval
import utils
from detector import Detector
import copy


NUM_CATEGORIES = 15
VALIDATION_ITERATION = 500
MAX_ITERATIONS = 15000
LEARNING_RATE = 1e-4
WEIGHT_POS = 2
WEIGHT_NEG = 1
WEIGHT_REG = 1
WEIGHT_CLASS = 8
BATCH_SIZE = 8

def train(device="cpu"):
    """Train the network.

    Args:
        device: The device to train on."""

    wandb.init(project="detector_baseline")

    # Init model
    detector = Detector().to(device)

    model_path = "./model.pt"
    utils.load_model(detector,model_path,device)


    dataset = CocoDetection(
        root="./dd2419_coco/training",
        annFile="./dd2419_coco/annotations/training.json",
        transforms=detector.input_transform,
    )

    val_dataset = CocoDetection(
        root="./dd2419_coco/validation",
        annFile="./dd2419_coco/annotations/validation.json",
        transforms=detector.input_transform_validation,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    current_iteration = 1
    while current_iteration <= MAX_ITERATIONS:
        validate(detector, val_dataloader, current_iteration, device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)


    print("Training started...")


def validate(detector, val_dataloader, current_iteration, device):
    detector.eval()
    coco_pred = copy.deepcopy(val_dataloader.dataset.coco)
    coco_pred.dataset["annotations"] = []
    with torch.no_grad():
        count = total_pos_mse = total_reg_mse = total_neg_mse = loss = 0
        image_id = ann_id = 0
        for val_img_batch, val_target_batch in val_dataloader:
            val_img_batch = val_img_batch.to(device)
            val_target_batch = val_target_batch.to(device)
            val_out = detector(val_img_batch)
            reg_mse, pos_mse, neg_mse, class_mse = compute_loss(val_out, val_target_batch)
            total_reg_mse += reg_mse
            total_pos_mse += pos_mse
            total_neg_mse += neg_mse
            loss += WEIGHT_POS * pos_mse + WEIGHT_REG * reg_mse + WEIGHT_NEG * neg_mse
            imgs_bbs = detector.decode_output(val_out, topk=100)
            for img_bbs in imgs_bbs:
                for img_bb in img_bbs:
                    coco_pred.dataset["annotations"].append(
                        {
                            "id": ann_id,
                            "bbox": [
                                img_bb["x"],
                                img_bb["y"],
                                img_bb["width"],
                                img_bb["height"],
                            ],
                            "area": img_bb["width"] * img_bb["height"],
                            "category_id": img_bb["category"], 
                            "score": img_bb["score"],
                            "image_id": image_id,
                        }
                    )
                    ann_id += 1
                image_id += 1
            count += len(val_img_batch) / BATCH_SIZE
        coco_pred.createIndex()
        coco_eval = COCOeval(val_dataloader.dataset.coco, coco_pred, iouType="bbox")
        coco_eval.params.useCats = 1 
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        print(
            {
                "total val loss": (loss / count),
                "val loss pos": (total_pos_mse / count),
                "val loss neg": (total_neg_mse / count),
                "val loss reg": (total_reg_mse / count),
                "val AP @IoU 0.5:0.95": coco_eval.stats[0],
                "val AP @IoU 0.5": coco_eval.stats[1],
                "val AR @IoU 0.5:0.95": coco_eval.stats[8],
            },
            step=current_iteration,
        )
        print(
            "Validation: {}, validation loss: {}".format(
                current_iteration, loss / count
            ),
        )
    detector.train()



def compute_loss(prediction_batch, target_batch):
    """Compute loss between predicted tensor and target tensor.
    Args:
        prediction_batch: Batched predictions. Shape (N,C,H,W).
        target_batch: Batched targets. shape (N,C,H,W).
    Returns:
        Tuple of separate loss terms:
            reg_mse:
            pos_mse:
            neg_mse:
    """
    # positive / negative indices
    # (this could be passed from input_transform to avoid recomputation)
    pos_indices = torch.nonzero(target_batch[:, 4, :, :] == 1, as_tuple=True)
    neg_indices = torch.nonzero(target_batch[:, 4, :, :] == 0, as_tuple=True)
    
    # compute loss
    reg_mse = nn.functional.mse_loss(
        prediction_batch[pos_indices[0], 0:4, pos_indices[1], pos_indices[2]],
        target_batch[pos_indices[0], 0:4, pos_indices[1], pos_indices[2]],
    )
    pos_mse = nn.functional.mse_loss(
        prediction_batch[pos_indices[0], 4, pos_indices[1], pos_indices[2]],
        target_batch[pos_indices[0], 4, pos_indices[1], pos_indices[2]],
    )
    neg_mse = nn.functional.mse_loss(
        prediction_batch[neg_indices[0], 4, neg_indices[1], neg_indices[2]],
        target_batch[neg_indices[0], 4, neg_indices[1], neg_indices[2]],
    )
    
    class_mse = nn.functional.mse_loss(
        prediction_batch[pos_indices[0], 5, pos_indices[1], pos_indices[2]],
        target_batch[pos_indices[0], 5, pos_indices[1], pos_indices[2]],
    )
    return reg_mse, pos_mse, neg_mse, class_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--cpu", dest="device", action="store_const", const="cpu")
    device.add_argument("--gpu", dest="device", action="store_const", const="cuda")
    args = parser.parse_args()
    train(args.device)
