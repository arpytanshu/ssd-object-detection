#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:17:07 2020

@author: ansh
"""

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image

def xywh_to_xyXY(boxes_xywh : torch.tensor) -> torch.tensor:
    '''
    Get bounding box coordinates in [ x_top_left, y_top_left, x_bottom_right, y_bottom_right] format.

    Parameters
    ----------
    boxes : Bounding Box, a tensor in [ x_top_left, y_top_left, bb_width, bb_height] format.

    '''
    boxes_xyXY = boxes_xywh.clone()
    boxes_xyXY[:,2] = boxes_xyXY[:,2] + boxes_xyXY[:,0]
    boxes_xyXY[:,3] = boxes_xyXY[:,3] + boxes_xyXY[:,1]
    return boxes_xyXY


def xyXY_to_xywh(boxes_xyXY : torch.tensor) -> torch.tensor:
    '''
    Get bounding box coordinates in [ x_top_left, y_top_left, bb_width, bb_height] format.

    Parameters
    ----------
    boxes_xywh : list or tensor in [ x_top_left, y_top_left, x_bottom_right, y_bottom_right] format.

    '''
    boxes_xywh = boxes_xyXY.clone()
    boxes_xywh[:,2] = boxes_xywh[:,2] - boxes_xywh[:,0]
    boxes_xywh[:,3] = boxes_xywh[:,3] - boxes_xywh[:,1]
    return boxes_xywh


def showBB_xyXY(image, boxes, scale = 300):
    if (type(image) == torch.Tensor):
        image = to_pil_image(image)
    plt.imshow(image)
    for bb in boxes:
        x,y,X,Y = map(int, bb*scale)
        plt.plot([x, X, X, x, x], [y, y, Y, Y, y], c='b', linewidth=1)
    plt.show()


def get_params_to_learn(model):
    params_to_learn = []
    for param in model.named_parameters():
        if param.requires_grad:
            params_to_learn.append(param)
    return params_to_learn

def get_mean_AR(df):
    samples = df.BB_xywh.values
    AR = list()
    for boxes in samples:
        for box in boxes:
            AR.append(box[2] / box[3])
    AR = np.array(AR)
    return AR.mean(), AR.std()      


def create_prior_boxes(self):
    """
    Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

    :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
    """
    import torch
    from math import sqrt
    
    fmap_dims = {'conv4_3': 38,
                 'conv7': 19,
                 'conv8_2': 10,
                 'conv9_2': 5,
                 'conv10_2': 3,
                 'conv11_2': 1}

    obj_scales = {'conv4_3': 0.1,
                  'conv7': 0.2,
                  'conv8_2': 0.375,
                  'conv9_2': 0.55,
                  'conv10_2': 0.725,
                  'conv11_2': 0.9}

    aspect_ratios = {'conv4_3': [1., 2., 0.5],
                     'conv7': [1., 2., 3., 0.5, .333],
                     'conv8_2': [1., 2., 3., 0.5, .333],
                     'conv9_2': [1., 2., 3., 0.5, .333],
                     'conv10_2': [1., 2., 0.5],
                     'conv11_2': [1., 2., 0.5]}

    fmaps = list(fmap_dims.keys())

    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                    # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                    # scale of the current feature map and the scale of the next feature map
                    if ratio == 1.:
                        try:
                            additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
    prior_boxes.clamp_(0, 1)  # (8732, 4)

    return prior_boxes