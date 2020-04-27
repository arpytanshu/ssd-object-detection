#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:17:07 2020

@author: ansh
"""
# -------
# Imports
# -------

import data
import utils
import torch
from random import randint
from torch.utils.data import DataLoader
from torchvision import transforms as T



# ---- -----
# Data Paths
# ---- -----
annotations_path = './data/annotation.txt'
image_path = './data/ShelfImages/'



# ---------- -------- -----------
# DataFrames DataSets DataLoaders
# ---------- -------- -----------

df = data.get_dataframe(annotations_path)
dataset = data.ShelfImageDataset(df, image_path)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=data.collate_fn)


#%%
# test dataset
# ---- -------
image, boxes = dataset[randint(0, len(dataset))]
utils.showBB_xyXY(image, boxes)

# test dataloader
# ---- ----------
batch = next(iter(dataloader))

images = batch[0]
boxes   = batch[1]

for image, bb in zip(images, boxes):
    utils.showBB_xyXY(image, bb)


#%%
# ---- ----------- -----
# test vggBackbone model
# ---- ----------- -----

from model.ssdconfig import SSDConfig
from model.ssd import VggBackbone
import torch

config = SSDConfig()
model = VggBackbone(config)

x = torch.rand(5, 3, 300, 300)

out = model(x)
for o in out:
    print(o.shape)
    
#%%
# ---- -------------- -----
# test predictionConv model
# ---- -------------- -----

import pickle
from model.ssdconfig import SSDConfig
from model.ssd import PredictionConv

with open('ref/out.pickle' , 'rb') as f:
    out = pickle.load(f)

config = SSDConfig()
pc_model = PredictionConv(config)

loc_out, clf_out = pc_model(out)
print(loc_out.shape)
print(clf_out.shape)

#%%

import torch
from model.ssd import SSD
from model.ssdconfig import SSDConfig

images = torch.rand((5, 3, 300, 300))

model = SSD(SSDConfig())

loc_out, clf_out = model(images)

print('Loc out shape:', loc_out.shape)
print('Clf out shape:', clf_out.shape)


#%%



