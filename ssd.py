#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:17:07 2020

@author: arpytanshu@gmail.com
"""

import torch
import torch.nn.functional as F

from torch import nn
from math import sqrt
from ssdconfig import SSDConfig
import ssdutils
from torchvision.models import vgg16_bn, vgg16


class VggBackbone(nn.Module):
    def __init__(self, config: SSDConfig):
        super().__init__()

        self.config = config
        self.conv_43_index = self.config.VGGBN_BASE_CONV43_INDEX if self.config.VGG_BN_FLAG else self.config.VGG_BASE_CONV43_INDEX

        self.vgg_base = nn.ModuleList(self._vgg_layers())
        self.aux_base = nn.ModuleList(self._aux_layers())
        self._init_aux_params()
        self._load_vgg_params()
        
    def _vgg_layers(self):
        cfg = self.config.VGG_BASE_CONFIG[str(self.config.INPUT_IMAGE_SIZE)]
        in_channels = self.config.VGG_BASE_IN_CHANNELS
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.config.VGG_BN_FLAG:
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
        
        # apply vgg up to conv4_3
        for ix in range(self.conv_43_index):
            x = self.vgg_base[ix](x)
        features.append(x)

        # apply vgg up to fc7
        for ix in range(self.conv_43_index, len(self.vgg_base)):
            x = self.vgg_base[ix](x)
        features.append(x)
        
        # apply auxiliary conv layers
        for ix in range(len(self.aux_base)):
            x = F.relu(self.aux_base[ix](x))
            if(ix % 2 == 1):
                features.append(x)
        return features

    def _load_vgg_params(self):
        vgg16_pt = vgg16_bn if self.config.VGG_BN_FLAG else vgg16  # pretrained vgg16 model
        views = self.config.VGG_BASE_CONV67_VIEWS
        subsample_factor = self.config.VGG_BASE_CONV67_SUBSAMPLE_FACTOR
        # get pre-trained parameters
        pretrained_params = vgg16_pt(self.config.USE_PRETRAINED_VGG).features.state_dict()
        pretrained_clfr_params = vgg16_pt(self.config.USE_PRETRAINED_VGG).classifier.state_dict()

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
                index = torch.arange(0, end=tensor.size(d), step=m[d], dtype=torch.long)
                tensor = tensor.index_select(dim=d, index = index)                       
        return tensor

    def _init_aux_params(self):
        for m in self.aux_base.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)



class PredictionConv(nn.Module):
    def __init__(self, config: SSDConfig):
        super().__init__()
        self.config = config
        self.num_channels = self.config.FM_NUM_CHANNELS
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
        for ix, fm_name in enumerate(self.config.FM_NAMES):
            localization_layers.append(
                nn.Conv2d(in_channels=self.num_channels[ix],
                          out_channels=self.num_priors[ix]*4,
                          kernel_size=3, padding=1))
        return localization_layers

    def _get_classification_convs(self):
        classification_layers = []
        for ix, fm_name in enumerate(self.config.FM_NAMES):
            classification_layers.append(
                nn.Conv2d(in_channels=self.num_channels[ix],
                          out_channels=self.num_classes * self.num_priors[ix],
                          kernel_size=3, padding=1))
        return classification_layers

    def _init_conv_layers(self):
        # xavier initialize conv layers
        for child in self.children():
            for m in child.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)


        
class SSD(nn.Module):
    def __init__(self, config: SSDConfig):
        super().__init__()
        self.config = config
        self.vgg_backbone = VggBackbone(config)
        self.pred_convs = PredictionConv(config)
        self.rescale_factor = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        self.priors_cxcy = self.create_prior_box().to(self.config.DEVICE)
        
    def forward(self, images):
        feature_maps = self.vgg_backbone(images)
        # Rescale conv4_3 after L2 norm
        norm = feature_maps[0].pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        feature_maps[0] = feature_maps[0] / norm  # (N, 512, 38, 38)
        feature_maps[0] = feature_maps[0] * self.rescale_factor  # (N, 512, 38, 38)
        # Get Prediction Convolutions
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
                if additional_scales != []:
                    PRIORS.append([cx, cy, additional_scales[ix], additional_scales[ix]])
                for a_r in fm_aspect_ratios[ix]:
                    width = scale * sqrt(a_r)
                    height = scale / sqrt(a_r)
                    PRIORS.append([cx, cy, width, height])
        PRIORS = torch.FloatTensor(PRIORS)
        PRIORS.clamp_(0,1)
        return PRIORS
    
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k=0):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = ssdutils.cxcy_to_xy(
                ssdutils.gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.config.NUM_CLASSES):
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
                overlap = ssdutils.find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score)).to(torch.bool).to(self.config.DEVICE)  # (n_qualified)
                
                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == True:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = False

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[~suppress])
                image_labels.append(torch.LongTensor((~suppress).sum().item() * [c]).to(self.config.DEVICE))
                image_scores.append(class_scores[~suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.config.DEVICE))
                image_labels.append(torch.LongTensor([0]).to(self.config.DEVICE))
                image_scores.append(torch.FloatTensor([0.]).to(self.config.DEVICE))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k > 0:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size
        


class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, config : SSDConfig):
        super(MultiBoxLoss, self).__init__()
        self.config = config
        self.priors_cxcy = priors_cxcy.to(self.config.DEVICE)
        self.priors_xy = ssdutils.cxcy_to_xy(priors_cxcy).to(self.config.DEVICE)
        self.threshold = config.MBL_threshold
        self.neg_pos_ratio = config.MBL_neg_pos_ratio
        self.alpha = config.MBL_alpha
        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.config.DEVICE)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.config.DEVICE)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            # n_objects = boxes[i].size(0)

            overlap = ssdutils.find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = ssdutils.cxcy_to_gcxgcy(ssdutils.xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(self.config.DEVICE)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss
