#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:23:46 2020

@author: arpytanshu@gmail.com
"""

import torch
from . import ssdconfig
import torchvision
from torch import nn



class VggBackbone(nn.Module):
    '''
    VGG16 with BN : Feature extractor for SSD
    '''
    
    def __init__(self, config : ssdconfig.SSDConfig):
        super().__init__()
        self.config = config
        
        self.vgg_base = nn.ModuleList(self._vgg_layers())
        self.aux_base = nn.ModuleList(self._aux_layers())
        self._init_aux_params()
        self._load_vgg_params()

    def _vgg_layers(self):
        '''
        

        Args:
            config (TYPE): DESCRIPTION.

        Returns:
            layers (TYPE): DESCRIPTION.

        '''
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
        '''
        

        Args:
            config (TYPE): DESCRIPTION.

        Returns:
            layers (TYPE): DESCRIPTION.

        '''
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
        
    
    def _load_vgg_params(self):
        '''

        '''
        bn_flag = self.config.VGG_BASE_BN  # if batch normalization is enabled or not
        views = self.config.VGG_BASE_CONV67_VIEWS
        subsample_factor = self.config.VGG_BASE_CONV67_SUBSAMPLE_FACTOR
        
        # get pre-trained parameters
        if bn_flag:
            pretrained_conv_params = torchvision.models.vgg16_bn(False).features.state_dict()
            pretrained_clfr_params = torchvision.models.vgg16_bn(False).classifier.state_dict()
        else:   
            pretrained_conv_params = torchvision.models.vgg16(False).features.state_dict()
            pretrained_clfr_params = torchvision.models.vgg16(False).classifier.state_dict()
        
        # reshape and subsample parameters for conv6 & conv7 layers
        for ix, param_name in enumerate(list(self.vgg_base.state_dict().keys())[-4:]):
            params = pretrained_clfr_params[list(pretrained_clfr_params.keys())[ix]].view(views[ix])
            params = self._subsample(params, subsample_factor[ix])
            pretrained_conv_params[param_name] = params
        
        # load pretrained parameteres into model
        res = self.vgg_base.load_state_dict(pretrained_conv_params, strict=False)
        assert(res.__repr__() == '<All keys matched successfully>'), \
            'Error Loading pretrained parameters'
            

    def _subsample(self, tensor, m):
        """
        downsample by keeping every 'm'th value.
    
        This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
    
        :param tensor: tensor to be decimated
        :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
        :return: decimated tensor
        """
        assert tensor.dim() == len(m)
        for d in range(tensor.dim()):
            if m[d] is not None:
                tensor = tensor.index_select(dim=d,
                                             index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())
        return tensor


    def _init_aux_params(self):
        for m in self.aux_base.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
