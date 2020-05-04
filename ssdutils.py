#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:33:44 2020

@author: arpytanshu@gmail.com
"""

import torch
import torch.nn.functional as F
from math import sqrt


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h

def xy_to_cxcy(xy):
    # xy is in scale invariant dimensions
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h

def cxcy_to_xy(cxcy):
    # x or y min is center - half(width)
    # x or y max is center + half(height)
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)
    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)
    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)
    return intersection / union  # (n1, n2)

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

def detect_objects(predicted_locs, predicted_scores, priors_cxcy, min_score, max_overlap):
    """
    Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

    For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

    :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, in gcxgcy coords
                            a tensor of dimensions (N, 8732, 4)
    :param predicted_scores: class scores for each of the encoded locations/boxes,
                            a tensor of dimensions (N, 8732, n_classes)
    :param min_score: minimum threshold for a box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
    :return: detections (boxes, labels, and scores),lists of length batch_size
    """
    batch_size = predicted_locs.size(0)
    n_priors = priors_cxcy.size(0)
    n_classes = predicted_scores.size(2)
    predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lists to store final predicted boxes, labels, and scores for all images
    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

    for i in range(batch_size):
        # Decode object coordinates from the form we regressed predicted boxes to
        decoded_locs = cxcy_to_xy(
            gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

        # Lists to store boxes and scores for this image
        image_boxes = list()
        image_labels = list()
        image_scores = list()

        max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

        # Check for each class
        for c in range(1, n_classes):
            # Keep only predicted boxes and scores where scores for this class are above the minimum score
            class_scores = predicted_scores[i][:, c]  # (8732)
            score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
            n_above_min_score = score_above_min_score.sum().item()
            if n_above_min_score == 0:
                continue
            class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
            class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

            # Sort predicted boxes and scores by scores
            class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
            class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

            # Find the overlap between predicted boxes
            overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

            # Non-Maximum Suppression (NMS)

            # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
            # 1 implies suppress, 0 implies don't suppress
            suppress = torch.zeros((n_above_min_score))  # (n_qualified)

            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If this box is already marked for suppression
                if suppress[box] == 1:
                    continue

                # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                # Find such boxes and update suppress indices
                flag = (overlap[box] > max_overlap).to(torch.float32)
                suppress = torch.max(suppress, flag )
                # The max operation retains previously suppressed boxes, like an 'OR' operation

                # Don't suppress this box, even though it has an overlap of 1 with itself
                suppress[box] = 0

            # Store only unsuppressed boxes for this class
            retained_box_index = (1-suppress).type(torch.bool)
            image_boxes.append(class_decoded_locs[retained_box_index])
            image_labels.append(torch.LongTensor(int((1 - suppress).sum().item()) * [c]))
            image_scores.append(class_scores[retained_box_index])

        # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
            image_labels.append(torch.LongTensor([0]).to(device))
            image_scores.append(torch.FloatTensor([0.]).to(device))

        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)

        # Append to lists that store predicted boxes and scores for all images
        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


def create_prior_box(config):
    fm_dims = config.FM_DIMS
    fm_names = config.FM_NAMES
    fm_scales = config.FM_SCALES
    fm_aspect_ratios = config.FM_ASPECT_RATIO
    additional_scales = config.FM_ADDITIONAL_SCALES
    PRIORS = list()
    for ix, fmap in enumerate(fm_names):
        dim = fm_dims[ix]
        scale = fm_scales[ix]
        for cx, cy in zip(torch.arange(dim).repeat(dim), torch.arange(dim).repeat_interleave(dim)):
            cx = (cx + 0.5) / dim
            cy = (cy + 0.5) / dim
            if additional_scales != []:
                    PRIORS.append([cx, cy, additional_scales[ix], additional_scales[ix]])
            for a_r in fm_aspect_ratios[ix]:
                width = scale * sqrt(a_r)
                height = scale / sqrt(a_r)
                PRIORS.append([cx, cy, width, height])
    PRIORS = torch.FloatTensor(PRIORS)
    PRIORS.clamp_(0,1)
    return PRIORS


def calc_mAP(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores):
    assert (len(gt_boxes) == len(gt_labels) == len(pred_boxes) == len(pred_labels) == len(pred_scores))
    
    gt_image_ix = []
    pred_image_ix = []
    
    for ix, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        assert(box.size(0) == label.size(0))
        gt_image_ix.extend([ix] * box.size(0))
    for ix, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        assert(box.size(0) == label.size(0) == score.size(0))
        pred_image_ix.extend([ix] * box.size(0))
    
    
    gt_image_ix = torch.tensor(gt_image_ix)
    gt_boxes = torch.cat(gt_boxes, dim=0)
    gt_labels = torch.cat(gt_labels, dim=0)
    
    pred_image_ix = torch.tensor(pred_image_ix)
    pred_boxes = torch.cat(pred_boxes, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_scores = torch.cat(pred_scores, dim=0)
    
    n_classes = gt_labels.unique().item() + 1
    AP = torch.zeros((n_classes - 1,), dtype=torch.float)
    
    for class_id in range(1, n_classes):
        # all ground truth box & label associated with this class ( across all the images )
        gt_ix = gt_labels == class_id
        gt_image_ix_class = gt_image_ix[gt_ix]
        gt_boxes_class = gt_boxes[gt_ix]
        gt_labels_class = gt_labels[gt_ix]
        
        # get all predictions for this class and sort them
        pred_ix = pred_labels == class_id
        _, pred_sorted_ix = torch.sort(pred_scores[pred_ix], dim=0, descending=True)
        
        pred_image_ix_class = pred_image_ix[pred_ix][pred_sorted_ix]
        pred_boxes_class = pred_boxes[pred_ix][pred_sorted_ix]
        pred_labels_class = pred_boxes[pred_ix][pred_sorted_ix]
        pred_scores_class = pred_scores[pred_ix][pred_sorted_ix]
        
        # number of detections for this class for this image
        n_class_detections = pred_boxes_class.size(0)
        if n_class_detections == 0: continue
    
        TP = torch.zeros((n_class_detections,))
        FP = torch.zeros((n_class_detections,))
        # To keep track gt_boxes that have already been detected.
        detected_gt_boxes = torch.zeros(gt_image_ix_class.size(0))
        
        for d in range(n_class_detections):
            this_image_ix = pred_image_ix_class[d]
            this_box = pred_boxes_class[d]
            
            # get all gt_boxes in this image which have the same class
            obj_same_class_in_image = gt_boxes_class[gt_image_ix_class == this_image_ix]
            # if no GT box exists for this class for this image, mark FP
            if obj_same_class_in_image.size(0) == 0:
                FP[d] = 1
                continue
            
            # find overlap of this detection with all gt boxes of the same class in this image
            overlap = find_jaccard_overlap(this_box.unsqueeze(0), obj_same_class_in_image)
            max_overlap, ind = torch.max(overlap.squeeze(0), dim=0)
            # index of box in gt_boxes_class with maximum overlap
            gt_matched_index = torch.LongTensor(range(gt_boxes_class.size(0)))[gt_image_ix_class == this_image_ix][ind]
            
            
            if max_overlap.item() > 0.5:
                # if this object has not already been detected, it's a TP
                if detected_gt_boxes[gt_matched_index] == 0:
                    TP[d] = 1
                    detected_gt_boxes[gt_matched_index] = 1 # this gt_box has been detected
                else:
                    FP[d] = 1
            else:
                FP[d] = 1
                
        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_TP = torch.cumsum(TP, dim=0)  # (n_class_detections)
        cumul_FP = torch.cumsum(FP, dim=0)  # (n_class_detections)
        cumul_precision = cumul_TP / (cumul_TP + cumul_FP + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_TP / n_class_detections  # (n_class_detections)
    
        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        AP[class_id - 1] = precisions.mean()  # c is in [1, n_classes - 1]
    
    mAP = AP.mean().item()    
    return AP, mAP