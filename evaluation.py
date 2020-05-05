#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:49:06 2020

@author: ansh
"""


import torch
from ssdconfig import SSDConfig
from data import ShelfImageDataset, collate_fn, get_dataframe
from torch.utils.data import DataLoader
from trainer import eval

config = SSDConfig()

df = get_dataframe(config.PATH_TO_ANNOTATIONS)
dataset = ShelfImageDataset(df, config.PATH_TO_IMAGES, train=False)
dataloader = DataLoader(dataset,
                        shuffle=True,
                        collate_fn=collate_fn,
                        batch_size=config.TRAIN_BATCH_SIZE,
                        num_workers=config.NUM_DATALOADER_WORKERS)


ckpt_path = './checkpoints/checkpoint_ssd_1.pth.tar'
# ckpt_path = 'checkpoints/checkpoint_ssd_2-1AP.pth.tar'
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
model = checkpoint['model']
model.config.DEVICE = torch.device('cpu')

mAP = eval(model, dataloader, min_score=0.5, max_overlap=0.5)

print(mAP)