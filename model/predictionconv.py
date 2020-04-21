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
    def __init__(self, config: SSDConfig):
        super().__init__()
        self.config = config
        self.num_channels = self.config.FEATURE_MAP_NUM_CHANNELS
        self.num_priors = self.config.NUM_PRIOR_PER_FM_CELL
        self.num_classes = self.config.NUM_CLASSES
        self.loc_conv = nn.ModuleList(self._get_localization_convs())
        self.clf_conv = nn.ModuleList(self._get_classification_convs())
        self._init_conv_layers()

    def forward(self, X):
        loc_out = []
        clf_out = []
        batch_size = X[1].shape[0]
        for ix, feature_map in enumerate(X):
            out1 = self.loc_conv[ix](feature_map)
            out2 = self.clf_conv[ix](feature_map)
            out1 = out1.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
            out2 = out2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
            loc_out.append(out1)
            clf_out.append(out2)
        loc_out = torch.cat(loc_out, dim=1)
        clf_out = torch.cat(clf_out, dim=1)
        return loc_out, clf_out

    def _get_localization_convs(self):
        localization_layers = []
        for fm_name in self.config.FEATURE_MAP_NAMES:
            localization_layers.append(
                nn.Conv2d(in_channels=self.num_channels[fm_name],
                          out_channels=self.num_priors[fm_name]*4,
                          kernel_size=3, padding=1))
        return localization_layers

    def _get_classification_convs(self):
        classification_layers = []
        for fm_name in self.config.FEATURE_MAP_NAMES:
            classification_layers.append(
                nn.Conv2d(in_channels=self.num_channels[fm_name],
                          out_channels=self.num_classes * self.num_priors[fm_name],
                          kernel_size=3, padding=1))
        return classification_layers

    def _init_conv_layers(self):
        # xavier initialize conv layers
        for child in self.children():
            for m in child.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
