#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:17:07 2020

@author: arpytanshu@gmail.com
"""

import torch
from torch import nn
from math import sqrt
from ssdconfig import SSDConfig
from vggbackbone import VggBackbone
from predictionconv import PredictionConv


class SSD(nn.Module):
    def __init__(self, config: SSDConfig):
        self.config = config
        self.vgg_backbone = VggBackbone(config)
        self.pred_convs = PredictionConv(config)
        self.rescale_factor = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        
    def forward(self, images):
        feature_maps = self.vgg_backbone(images)
        # Rescale conv4_3 after L2 norm
        norm = feature_maps[0].pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        feature_maps[0] = feature_maps[0] / norm  # (N, 512, 38, 38)
        feature_maps[0] = feature_maps[0] * self.rescale_factors  # (N, 512, 38, 38)
        loc_preds, clf_preds = self.pred_convs(feature_maps)
        return loc_preds, clf_preds

    def create_prior_box(self):
        fm_dims = self.config.FM_DIMS
        fm_names = self.config.FM_NAMES
        fm_scales = self.config.FM_SCALES
        fm_aspect_ratios = self.config.FM_ASPECT_RATIO
        additional_scales = self.config.FM_ADDITIONAL_SCALES
        PRIORS = list()
        for ix, fmap in enumerate(fm_names):
            dim = fm_dims[ix]
            scale = fm_scales[ix]
            for cx, cy in zip(torch.arange(dim).repeat(dim), torch.arange(dim).repeat_interleave(dim)):
                cx = (cx + 0.5) / dim
                cy = (cy + 0.5) / dim
                PRIORS.append([cx, cy, additional_scales[ix], additional_scales[ix]])
                for a_r in fm_aspect_ratios[ix]:
                    width = scale * sqrt(a_r)
                    height = scale / sqrt(a_r)
                    PRIORS.append([cx, cy, width, height])
        PRIORS = torch.FloatTensor(PRIORS)
        PRIORS.clamp_(0,1)
        return PRIORS
    
    def detect_objects(self):
                
        
              