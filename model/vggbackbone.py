#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:23:46 2020

@author: arpytanshu@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg16_bn, vgg16

from . import ssdconfig


class VggBackbone(nn.Module):
    def __init__(self, config: ssdconfig.SSDConfig):
        super().__init__()
        self.config = config
        self.vgg_base = nn.ModuleList(self._vgg_layers())
        self.aux_base = nn.ModuleList(self._aux_layers())
        self._init_aux_params()
        self._load_vgg_params()

    def _vgg_layers(self):
        cfg = self.config.VGG_BASE_CONFIG[str(self.config.INPUT_IMAGE_SIZE)]
        batch_norm = self.config.VGG_BASE_BN
        in_channels = self.config.VGG_BASE_IN_CHANNELS
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers

    def _aux_layers(self):
        # Extra layers added to VGG for feature scaling
        cfg = self.config.AUX_BASE_CONFIG[str(self.config.INPUT_IMAGE_SIZE)]
        in_channels = self.config.AUX_BASE_IN_CHANNELS
        layers = []
        flag = False
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'S':
                    layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        if self.config.INPUT_IMAGE_SIZE == 512:
            layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
            layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
        return layers

    def forward(self, x):
        features = []
        conv_43_index = self.config.VGGBN_BASE_CONV43_INDEX
        
        # apply vgg up to conv4_3
        for ix in range(conv_43_index):
            x = self.vgg_base[ix](x)
        # s = self.l2_norm(x)  # Conv4_3 L2 normalization # TODO : implement
        features.append(x)
        
        # apply vgg up to fc7
        for ix in range(conv_43_index, len(self.vgg_base)):
            x = self.vgg_base[ix](x)
        features.append(x)
        
        # apply auxiliary conv layers
        for ix in range(len(self.aux_base)):
            x = F.relu(self.aux_base[ix](x))
            if(ix % 2 == 1):
                features.append(x)
        return tuple(features)

    def _load_vgg_params(self):
        vgg16_pt = vgg16_bn if self.config.VGG_BASE_BN else vgg16  # pretrained vgg16 model
        views = self.config.VGG_BASE_CONV67_VIEWS
        subsample_factor = self.config.VGG_BASE_CONV67_SUBSAMPLE_FACTOR
        # get pre-trained parameters
        pretrained_params = vgg16_pt(False).features.state_dict()  # TODO : set to True
        pretrained_clfr_params = vgg16_pt(False).classifier.state_dict()  # TODO : set to True

        # reshape and subsample parameters for conv6 & conv7 layers
        # add reshaped classifier parameters to pretrained_params
        for ix, param_name in enumerate(list(self.vgg_base.state_dict().keys())[-4:]):
            params = pretrained_clfr_params[list(pretrained_clfr_params.keys())[ix]].view(views[ix])
            params = self._subsample(params, subsample_factor[ix])
            pretrained_params[param_name] = params
            
        # load pretrained parameteres into model
        res = self.vgg_base.load_state_dict(pretrained_params, strict=False)
        assert(res.__repr__() == '<All keys matched successfully>'), \
            'Error Loading pretrained parameters'

    def _subsample(self, tensor, m):
        # subsample a tensor by keeping every m-th value along a dimension
        # None for no subsampling in that dimension.
        assert tensor.dim() == len(m), \
            'Subsampling factor must be provided for each tensor dimension explicitly.'
        for d in range(tensor.dim()):
            if m[d] is not None:
                index = torch.arange(end=tensor.size(d), step=m[d], dtype=torch.long)
                tensor = tensor.index_select(dim=d, index = index)                       
        return tensor

    def l2_norm(self):
        # TODO : Implement
        pass

    def _init_aux_params(self):
        for m in self.aux_base.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
