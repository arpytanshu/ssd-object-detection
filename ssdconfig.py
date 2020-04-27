#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:17:07 2020

@author: arpytanshu@gmail.com
"""
import torch


class SSDConfig():
    def __init__(self):
        
        # Paths
        self.PATH_TO_ANNOTATIONS = './data/annotation.txt'
        self.PATH_TO_IMAGES = './data/ShelfImages/'
        self.PATH_TO_CHECKPOINT = ''
        
        # ------
        # device
        # ------
        self.DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # -----
        # Input
        # -----
        self.INPUT_IMAGE_SIZE = 300
        self.NUM_CLASSES = 2
        # -------------- --- --- --------
        # Configurations for VGG Backbone
        # -------------- --- --- --------
        self.VGG_BASE_BN = True
        self.VGG_BASE_IN_CHANNELS = 3
        self.VGG_BASE_CONFIG = {
            '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                    'C', 512, 512, 512, 'M', 512, 512, 512],
            '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                    'C', 512, 512, 512, 'M', 512, 512, 512]}
        # reshaped parameters dimensions for VGG's conv 6 & 7
        self.VGG_BASE_CONV67_VIEWS = [[4096, 512, 7, 7],
                                      [4096],
                                      [4096, 4096, 1, 1],
                                      [4096]]
        # subsampling ratio for VGG's conv 6 & 7
        self.VGG_BASE_CONV67_SUBSAMPLE_FACTOR = [[4, None, 3, 3],
                                                 [4],
                                                 [4, 4, None, None],
                                                 [4]]
        self.VGGBN_BASE_CONV43_INDEX = 33
        self.VGG_BASE_CONV43_INDEX = None  # TODO : find value
        # -------------- --- ---------- ------------
        # Configurations for Auxiliary convolutions
        # -------------- --- ---------- ------------
        self.AUX_BASE_IN_CHANNELS = 1024
        self.AUX_BASE_CONFIG = {
            '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
            '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256]}
        # ------- ----
        # Feature Maps
        # ------- ----
        self.FM_NAMES = ['conv4_3', 'conv7', 'conv8_2',
                        'conv9_2', 'conv10_2', 'conv11_2']
        self.FM_NUM_CHANNELS = {
            'conv4_3': 512,
            'conv7': 1024,
            'conv8_2': 512,
            'conv9_2': 256,
            'conv10_2': 256,
            'conv11_2': 256}
        self.NUM_PRIOR_PER_FM_CELL = {
            'conv4_3': 4,
            'conv7': 6,
            'conv8_2': 6,
            'conv9_2': 6,
            'conv10_2': 4,
            'conv11_2': 4}
        self.FM_DIMS = [38, 19, 10, 5, 3, 1]
        self.FM_SCALES = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
        self.FM_ASPECT_RATIO = [[1., 2., 0.5],
                                [1., 2., 3., 0.5, .333],
                                [1., 2., 3., 0.5, .333],
                                [1., 2., 3., 0.5, .333],
                                [1., 2., 0.5],
                                [1., 2., 0.5]]
        # ith additional scale is geometric mean scales of ith and (i+1)th FM.
        self.FM_ADDITIONAL_SCALES = [0.1414, 0.2738, 0.4541, 0.6314, 0.8077, 1.0]
        
        # ---- -------------
        # Loss configuration
        # ---- -------------
        self.MBL_threshold = 0.5
        self.MBL_neg_pos_ratio = 3
        self.MBL_alpha = 1.
        
        # -------- ------------
        # Training configration
        # -------- ------------
        self.TRAIN_BATCH_SIZE = 4
        self.NUM_DATALOADER_WORKERS = 1  # 
        self.NUM_ITERATIONS_TRAIN = 10000  # number of iterations to train
        self.LEARNING_RATE = 0.001
        self.DECAY_LR_AT = [7500, 8500] # number of times to decay the LR by DECAY_FRAC
        self.DECAY_FRAC = 0.1  # decay LR by this fraction of the current learning rate
        self.WEIGHT_DECAY = 5e-4
        self.MOMENTUM = 0.9
        self.PRINT_FREQ = 20
        