#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:17:07 2020

@author: ansh
"""

import torch
import torchvision
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

