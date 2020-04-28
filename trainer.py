#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:17:07 2020

@author: arpytanshu@gmail.com
"""

import time
import utils
import torch
from ssdconfig import SSDConfig
from data import ShelfImageDataset, collate_fn, get_dataframe
from torch.utils.data import DataLoader
from torch.optim import SGD
from ssd import SSD, MultiBoxLoss

config = SSDConfig()
device = config.DEVICE


def main():
    try:
        checkpoint = torch.load(config.PATH_TO_CHECKPOINT)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    except FileNotFoundError:
        print('PATH_TO_CHECKPOINT not specified in SSDConfig.\nMaking new model and optimizer.')
        start_epoch = 0
        model = SSD(config)
        model_parameters = utils.get_model_params(model)
        optimizer = SGD(params=[{'params': model_parameters['biases'], 'lr': 2 * config.LEARNING_RATE},
                            {'params': model_parameters['not_biases']}],
                            lr=config.LEARNING_RATE,
                            momentum=config.MOMENTUM,
                            weight_decay=config.WEIGHT_DECAY)
 
    # dataloader
    df = get_dataframe(config.PATH_TO_ANNOTATIONS)
    dataset = ShelfImageDataset(df, config.PATH_TO_IMAGES, train=True)
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            collate_fn=collate_fn,
                            batch_size=config.TRAIN_BATCH_SIZE,
                            num_workers=config.NUM_DATALOADER_WORKERS)
    
    # move to device
    model.to(device)
    criterion = MultiBoxLoss(model.priors_cxcy, config).to(device)
    # num epochs to train
    epochs = config.NUM_ITERATIONS_TRAIN // len(dataloader)
    # fooh!!!! :)
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
