#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:14:10 2020

@author: arpytanshu@gmail.com
"""


class SSDConfig():
    def __init__(self):
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
        self.FEATURE_MAP_NAMES = ['conv4_3', 'conv7', 'conv8_2',
                                  'conv9_2', 'conv10_2', 'conv11_2']
        self.FEATURE_MAP_NUM_CHANNELS = {
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
