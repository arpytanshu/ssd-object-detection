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
from tqdm import tqdm
from ssdutils import calc_mAP

#%%

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
model.eval()


#%%

min_score = 0.4
max_overlap = 0.4

gt_boxes = []
gt_labels = []
pred_boxes = []
pred_labels = []
pred_scores = []

for batch in tqdm(dataloader):
    
    orig_images = batch[0]
    orig_boxes = batch[1]
    orig_labels = batch[2]
    
    
    with torch.no_grad():
        loc_gcxgcy, scores = model(orig_images)
        boxes, labels, scores = model.detect_objects(loc_gcxgcy, scores, min_score=min_score, max_overlap=max_overlap)

    gt_boxes.extend(orig_boxes)
    gt_labels.extend(orig_labels)
    pred_boxes.extend(boxes)
    pred_labels.extend(labels)
    pred_scores.extend(scores)


AP, mAP = calc_mAP(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores)
print(mAP)


