#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:14:10 2020

@author: arpytanshu@gmail.com
"""

class SSDConfig():
    def __init__(self):
        self.INPUT_IMAGE_SIZE = 300
        
        self.VGG_BASE_BN = True
        self.VGG_BASE_IN_CHANNELS = 3
        self.VGG_BASE_CONFIG = {
            '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
            '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
            }
        self.VGG_BASE_CONV67_VIEWS = [[4096, 512, 7, 7], [4096], [4096, 4096, 1, 1], [4096]]
        self.VGG_BASE_CONV67_SUBSAMPLE_FACTOR = [[4, None, 3, 3], [4], [4, 4, None, None], [4]]


        self.AUX_BASE_IN_CHANNELS = 1024
        self.AUX_BASE_CONFIG = {
            '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
            '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
            }
