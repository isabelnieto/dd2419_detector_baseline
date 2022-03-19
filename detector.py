"""Baseline detector model.
Inspired by
You only look once: Unified, real-time object detection, Redmon, 2016.
"""
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import utils
from torchvision import models
from torchvision import transforms


class Detector(nn.Module):
    """Baseline module for object detection."""

    def __init__(self):
        """Create the module.
        Define all trainable layers.
        """
        super(Detector, self).__init__()

        self.features = models.mobilenet_v2(pretrained=True).features
        # output of mobilenet_v2 will be 1280x15x20 for 480x640 input images

        self.head = nn.Conv2d(in_channels=1280, out_channels=6, kernel_size=1)
        # 1x1 Convolution to reduce channels to out_channels without changing H and W

        # 1280x15x20 -> 5x15x20, where each element 5 channel tuple corresponds to
        #   (rel_x_offset, rel_y_offset, rel_x_width, rel_y_height, confidence, id)
        # Where rel_x_offset, rel_y_offset is relative offset from cell_center
        # Where rel_x_width, rel_y_width is relative to image size
        # Where confidence is predicted IOU * probability of object center in this cell
        # Where id is the category of the traffic sign
        self.out_cells_x = 20.0
        self.out_cells_y = 15.0
        self.img_height = 480.0
        self.img_width = 640.0

    def forward(self, inp):
        """Forward pass.
        Compute output of neural network from input.
        """
        features = self.features(inp)
        out = self.head(features)  # Linear (i.e., no) activation

        return out

    def decode_output(self, out, threshold):
        """Convert output to list of bounding boxes.
        Args:
            out (torch.tensor):
                The output of the network.
                Shape expected to be NxCxHxW with
                    N = batch size
                    C = channel size
                    H = image height
                    W = image width
            threshold (float):
                The threshold above which a bounding box will be accepted.
        Returns:
            List[List[Dict]]
            List containing a list of detected bounding boxes in each image.
            Each dictionary contains the following keys:
                - "x": Top-left corner column
                - "y": Top-left corner row
                - "width": Width of bounding box in pixel
                - "height": Height of bounding box in pixel
                - "category": Category (not implemented yet!)
        """
        bbs = []
        # decode bounding boxes for each image
        for o in out:
            img_bbs = []

            # find cells with bounding box center
            bb_indices = torch.nonzero(o[4, :, :] >= threshold)

            # loop over all cells with bounding box center
            for bb_index in bb_indices:
                bb_coeffs = o[0:5, bb_index[0], bb_index[1]]

                # decode bounding box size and position
                width = self.img_width * bb_coeffs[2]
                height = self.img_height * bb_coeffs[3]
                y = (
                    self.img_height / self.out_cells_y * (bb_index[0] + bb_coeffs[1])
                    - height / 2.0
                )
                x = (
                    self.img_width / self.out_cells_x * (bb_index[1] + bb_coeffs[0])
                    - width / 2.0
                )
                category = o[5, bb_index[0], bb_index[1]]
                
                
                img_bbs.append(
                    {
                        "width": width,
                        "height": height,
                        "x": x,
                        "y": y,
                        "category": category,
                    }
                )
            bbs.append(img_bbs)

        return bbs

    def input_transform(self, image, anns):
        """Prepare image and targets on loading.
        This function is called before an image is added to a batch.
        Must be passed as transforms function to dataset.
        Args:
            image (PIL.Image):
                The image loaded from the dataset.
            anns (List):
                List of annotations in COCO format.
        Returns:
            Tuple:
                - (torch.Tensor) The image.
                - (torch.Tensor) The network target containing the bounding box.
        """
        #Transfor color
        cj = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        image = cj(image)

        #Transform position
        
        ra = transforms.RandomAffine(0, [0.1, 0.1])
        angle, translations, scale, shear = ra.get_params(ra.degrees, ra.translate, ra.scale, ra.shear, image.size)
        TF.affine(image, angle, translations, scale, shear, resample=ra.resample, fillcolor=ra.fillcolor,)
        

        for ann in anns:
            ann["bbox"][0] += translations[0]
            ann["bbox"][1] += translations[1]
            #check is inside the limits
            x_pos = ann["bbox"][0] + ann["bbox"][2]/2
            x_neg = ann["bbox"][0] - ann["bbox"][2]/2
            y_pos = ann["bbox"][1] + ann["bbox"][3]/2
            y_neg = ann["bbox"][1] - ann["bbox"][3]/2

            if x_pos > 640:
                 ann["bbox"][0] = 640
            
            if x_neg < 0:
                 ann["bbox"][0] = 0
            
            if x_pos > 480:
                 ann["bbox"][1] = 480
            
            if x_neg < 0:
                 ann["bbox"][1] = 0

        

        # Convert PIL.Image to torch.Tensor
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image)
        
        # Convert bounding boxes to target format
        
        # First two channels contain relativ x and y offset of bounding box center
        # Channel 3 & 4 contain relative width and height, respectively
        # Last channel is 1 for cell with bounding box center and 0 without

        # If there is no bb, the first 4 channels will not influence the loss
        # -> can be any number (will be kept at 0)
        target = torch.zeros(6, 15, 20)
        for ann in anns:
            x = ann["bbox"][0]
            y = ann["bbox"][1]
            width = ann["bbox"][2]
            height = ann["bbox"][3]
            category_id = ann["category_id"]

            x_center = x + width / 2.0
            y_center = y + height / 2.0
            x_center_rel = x_center / self.img_width * self.out_cells_x
            y_center_rel = y_center / self.img_height * self.out_cells_y
            x_ind = int(x_center_rel)
            y_ind = int(y_center_rel)
            x_cell_pos = x_center_rel - x_ind
            y_cell_pos = y_center_rel - y_ind
            rel_width = width / self.img_width
            rel_height = height / self.img_height
            rel_id = (category_id+1)

            # channels, rows (y cells), cols (x cells)
            target[4, y_ind, x_ind] = 1

            # bb size
            target[0, y_ind, x_ind] = x_cell_pos
            target[1, y_ind, x_ind] = y_cell_pos
            target[2, y_ind, x_ind] = rel_width
            target[3, y_ind, x_ind] = rel_height
            target[5, y_ind, x_ind] = rel_id

        return image, target
