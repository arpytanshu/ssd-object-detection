#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 01:06:53 2020

@author: arpytanshu@gmail.com
"""

import torch
from torch import nn
from model.ssdconfig import SSDConfig

class PredictionConv(nn.Module):
    
    def __init__(self, config : SSDConfig):
        self.config = config

    def _get_localization_convs(self):
        localization_layers = []
        num_channels = self.config.FEATURE_MAP_NUM_CHANNELS
        num_priors = self.config.NUM_PRIOR_PER_FM_CELL
        for fm_name in self.config.FEATURE_MAP_NAMES:
            localization_layers.append(nn.Conv2d(in_channels = num_channels[fm_name],
                                                 out_channels = num_priors[fm_name]*4,
                                                 kernel_size = 3, padding = 1))
        return localization_layers
    
    def _get_classification_convs(self):
        classification_layers = []
        num_channels = self.config.FEATURE_MAP_NUM_CHANNELS
        num_classes = self.config.NUM_CLASSES
        for fm_name in self.config.FEATURE_MAP_NAMES:
            classification_layers.append(nn.Conv2d(in_channels = num_channels[fm_name],
                                                   out_channels = num_classes,
                                                   kernel_size = 3, padding = 1))
        return classification_layers
    
        
    