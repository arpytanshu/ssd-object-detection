#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:17:07 2020

@author: ansh
"""

from PIL import Image
from os import listdir
from random import uniform
from torch.utils.data import Dataset


import utils
import torch
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as FT

# -------------- ---------
# Transformation Functions
# -------------- ---------

def resize(image, boxes, dims=(300,300)):
    '''
    Resize an image and bounding boxes.
    Args:
        image (PIL.Image)
        boxes (Tensor): A tensor of dimensions (n_objects, 4)
                representing the bounding boxes
        dims (Tuple): Output image size. Defaults to (300,300).
    Returns:
        Resized PIL image, updated bounding box coordinates.
    '''
    # resize image
    new_image = FT.resize(image, dims)
    # resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates 
    return new_image, new_boxes


def hflip(image, boxes):
    '''
    Horizontally flip an image and bounding boxes.
    
    Parameters
    ----------
    image : a PIL Image
    boxes : a tensor of dimensions (n_objects, 4)
        Bounding box in [ x_min, y_min, x_max, y_max ] format.
    Returns
    -------
    flipped image, updated bounding box coordinates.
    '''
    # Flip image
    new_image = FT.hflip(image)
    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    return new_image, new_boxes


def imageTransforms(image):
    '''
    Applies following transforms to images:
        - Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
        - Converts PIL image to torch tensor
        - Normalizes image tensor as expected by pre-trained torchvision models
    
    Parameters
    ----------
    image : a PIL Image

    Returns
    -------
    Transformed image tensor.

    '''
    ImageTransforms = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    new_image = ImageTransforms(image)
    return new_image

# ------- -------- ---------
# Dataset specific functions    
# ------- -------- ---------

def get_dataframe(annotations_path):
    D =    {'image_name' : [],
            'num_objects': [],
            'BB_xywh': [],
            'object_ids' : []}    
    for line in open(annotations_path, 'r'):
        image_name, num_objects, *annotations = line.split()
        annotations_list = [list(map(int, annotations[i:i+4])) for i in range(0, len(annotations), 5)]
        object_id_list = [int(annotations[i+4]) for i in range(0, len(annotations), 5)]
        D['image_name'].append(image_name)
        D['num_objects'].append(num_objects)
        D['BB_xywh'].append(annotations_list)
        D['object_ids'].append(object_id_list)    
    df = pd.DataFrame(D)
    return df


def collate_fn(batch):
    images = list()
    boxes = list()
    labels = list()
    for b in batch:
        images.append(b[0])
        boxes.append(b[1])
        labels.append(b[2])
    images = torch.stack(images, dim=0)
    return images, boxes, labels
   
   
class ShelfImageDataset(Dataset):
    def __init__(self, df, image_path, train=True, return_orig=False):
        self.df = df
        self.train = train
        self.return_orig = return_orig
        self.image_path = image_path+'train/' if train else image_path+'test/'
        self._fix_df()
        
    def __len__(self):
        return len(self.df)
    
    def _fix_df(self):
        list_images = listdir(self.image_path)
        self.df = self.df[self.df.image_name.isin(list_images)]
        self.df.reset_index(drop=True, inplace=True)
        
    def __getitem__(self, idx):
        orig_image = Image.open(self.image_path + self.df.loc[idx, 'image_name']).convert('RGB')
        boxes = torch.tensor(self.df.loc[idx, 'BB_xywh'])
        boxes = utils.xywh_to_xyXY(boxes)
        # rescale image and boxes
        image, boxes = resize(orig_image, boxes, (300,300))
        # horizontal flip image and boxes with 50% prob (if it's a train sample)
        if (uniform(0,1)>0.5 and self.train): 
            image, boxes = hflip(image, boxes)
        image = imageTransforms(image)
        label = torch.LongTensor([1]*boxes.size(0))
        if self.return_orig:
            return image, boxes, label, orig_image
        else:
            return image, boxes, label
