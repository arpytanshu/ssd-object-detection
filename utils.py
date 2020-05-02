#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:17:07 2020

@author: ansh
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, model, optimizer, config, path):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'config': config}
    torch.save(state, path)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def xywh_to_xyXY(boxes_xywh : torch.tensor) -> torch.tensor:
    '''
    Get bounding box coordinates in [ x_top_left, y_top_left, x_bottom_right, y_bottom_right] format.

    Parameters
    ----------
    boxes : Bounding Box, a tensor in [ x_top_left, y_top_left, bb_width, bb_height] format.

    '''
    boxes_xyXY = boxes_xywh.clone()
    boxes_xyXY[:,2] = boxes_xyXY[:,2] + boxes_xyXY[:,0]
    boxes_xyXY[:,3] = boxes_xyXY[:,3] + boxes_xyXY[:,1]
    return boxes_xyXY


def showBB_xyXY(image, boxes, scale = 300):
    if (type(image) == torch.Tensor):
        image = to_pil_image(image)
    plt.imshow(image)
    for bb in boxes:
        x,y,X,Y = map(int, bb*scale)
        plt.plot([x, X, X, x, x], [y, y, Y, Y, y], c='b', linewidth=1)
    plt.show()


def get_params_to_learn(model):
    params_to_learn = []
    for param in model.named_parameters():
        if param.requires_grad:
            params_to_learn.append(param)
    return params_to_learn


def get_mean_AR(df):
    samples = df.BB_xywh.values
    AR = list()
    for boxes in samples:
        for box in boxes:
            AR.append(box[2] / box[3])
    AR = np.array(AR)
    return AR.mean(), AR.std()      


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