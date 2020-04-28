#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:17:07 2020

@author: arpytanshu@gmail.com
"""

import time
import utils 
from ssdconfig import SSDConfig
from data import ShelfImageDataset, collate_fn, get_dataframe
from torch.utils.data import DataLoader
from torch.optim import SGD
from ssd import SSD, MultiBoxLoss

config = SSDConfig()
device = config.DEVICE

def get_model_params(model):
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    return {'biases' : biases, 'not_biases' : not_biases}


def main():

    global config, device
    
    # dataloader
    df = get_dataframe(config.PATH_TO_ANNOTATIONS)
    dataset = ShelfImageDataset(df, config.PATH_TO_IMAGES)
    dataloader = DataLoader(dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    # model
    model = SSD(config)
    model_parameters = get_model_params(model)
    model.to(device)
    # optimizer
    optimizer = SGD(params=[{'params': model_parameters['biases'], 'lr': 2 * config.LEARNING_RATE},
                            {'params': model_parameters['not_biases']}],
                            lr=config.LEARNING_RATE,
                            momentum=config.MOMENTUM,
                            weight_decay=config.WEIGHT_DECAY)
    # criterion
    criterion = MultiBoxLoss(model.priors_cxcy, config)
    criterion = criterion.to(device)
    
    epochs = config.NUM_ITERATIONS_TRAIN // len(dataloader)
    
    for epoch in range(epochs):
        if epoch in config.DECAY_LR_AT:
            utils.adjust_learning_rate(optimizer, config.DECAY_FRAC)
        train(dataloader, model, criterion, optimizer, epoch)
        utils.save_checkpoint(epoch, model, optimizer, config.PATH_TO_CHECKPOINT)
    
    
    
def train(dataloader, model, criterion, optimizer, epoch):

    model.train()  # training mode enables dropout

    batch_time = utils.AverageMeter()  # forward prop. + back prop. time
    data_time = utils.AverageMeter()  # data loading time
    losses = utils.AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, batch in enumerate(dataloader):
        data_time.update(time.time() - start)

        # Move to default device
        images = batch[0].to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in batch[1]]
        labels = [l.to(device) for l in batch[2]]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        # if grad_clip is not None:
        #     clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % config.PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(dataloader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
