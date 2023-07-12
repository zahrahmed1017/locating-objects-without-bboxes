__copyright__ = \
"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Javier Ribera, David Guera, Yuhao Chen, Edward J. Delp"
__version__ = "1.6.0"


import math
import torch
from sklearn.utils.extmath import cartesian
import numpy as np
from torch.nn import functional as F
import os
import time
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors.kde import KernelDensity
import skimage.io
from matplotlib import pyplot as plt
from torch import nn


torch.set_default_dtype(torch.float32)


def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"


def cdist(x, y):
    """
    Compute distance between each pair of the two collections of inputs.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||

    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances


def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
    between two unordered sets of points (the function is symmetric).
    Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    :return: The Averaged Hausdorff Distance between set1 and set2.
    """

    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    set1 = np.array(set1)
    set2 = np.array(set2)

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
        np.average(np.min(d2_matrix, axis=1))

    return res


class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).
        Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """

        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = cdist(set1, set2)

        # Modified Chamfer Loss
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term_1 + term_2

        return res


class WeightedHausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height, resized_width,
                 p=-9,
                 return_2_terms=False,
                 device=torch.device('cpu')):
        """
        :param resized_height: Number of rows in the image.
        :param resized_width: Number of columns in the image.
        :param p: Exponent in the generalized mean. -inf makes it the minimum.
        :param return_2_terms: Whether to return the 2 terms
                               of the WHD instead of their sum.
                               Default: False.
        :param device: Device where all Tensors will reside.
        """
        super(nn.Module, self).__init__()

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.max_dist = math.sqrt(resized_height**2 + resized_width**2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = self.all_img_locations.to(device=device,
                                                           dtype=torch.get_default_dtype())

        self.return_2_terms = return_2_terms
        self.p = p

    def forward(self, prob_map, gt, orig_sizes):
        """
        Compute the Weighted Hausdorff Distance function
        between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.

        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :param orig_sizes: Bx2 Tensor containing the size
                           of the original images.
                           B is batch size.
                           The size must be in (height, width) format.
        :param orig_widths: List of the original widths for each image
                            in the batch.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance.
        """

        _assert_no_grad(gt)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s'\
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):

            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b]
            orig_size_b = orig_sizes[b, :]
            norm_factor = (orig_size_b/self.resized_size).unsqueeze(0)
            n_gt_pts = gt_b.size()[0]

            # Corner case: no GT points
            if gt_b.ndimension() == 1 and (gt_b < 0).all().item() == 0:
                terms_1.append(torch.tensor([0],
                                            dtype=torch.get_default_dtype()))
                terms_2.append(torch.tensor([self.max_dist],
                                            dtype=torch.get_default_dtype()))
                continue

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1) *\
                self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1)*gt_b
            d_matrix = cdist(normalized_x, normalized_y)

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * \
                torch.sum(p * torch.min(d_matrix, 1)[0])
            weighted_d_matrix = (1 - p_replicated)*self.max_dist + p_replicated*d_matrix
            minn = generaliz_mean(weighted_d_matrix,
                                  p=self.p,
                                  dim=0, keepdim=False)
            term_2 = torch.mean(minn)

            # terms_1[b] = term_1
            # terms_2[b] = term_2
            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        if self.return_2_terms:
            res = terms_1.mean(), terms_2.mean()
        else:
            res = terms_1.mean() + terms_2.mean()

        return res


def generaliz_mean(tensor, dim, p=-9, keepdim=False):
    # """
    # Computes the softmin along some axes.
    # Softmin is the same as -softmax(-x), i.e,
    # softmin(x) = -log(sum_i(exp(-x_i)))

    # The smoothness of the operator is controlled with k:
    # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

    # :param input: Tensor of any dimension.
    # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    # :param keepdim: (bool) Whether the output tensor has dim retained or not.
    # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
    # """
    # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
    """
    The generalized mean. It corresponds to the minimum when p = -inf.
    https://en.wikipedia.org/wiki/Generalized_mean
    :param tensor: Tensor of any dimension.
    :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    :param keepdim: (bool) Whether the output tensor has dim retained or not.
    :param p: (float<0).
    """
    assert p < 0
    res= torch.mean((tensor + 1e-6)**p, dim, keepdim=keepdim)**(1./p)
    return res

def create_target_states(conf_pred,loc_pred,target_locations):
    """
    Inputs:
        conf_pred - tensor of shape (B,H,W) with values that are the confidence of each prediction
        loc_pred  - tensor of shape (B,H,W,2) with values that are x (index 0) and y (index 1) locations of the predictions 
        target_locations - List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (x,y), i.e, (col,row) of a GT point.
    Outputs: 
        target_states - a tensor of shape (B,H,W) that has a 1 for every pixel (H,W) that has an object
        target_locations_rsz - a tensor of shape (B,H,W,2) that has the normalized (0 to 1) location of the ground truth objects in the pixel
    """
    # Detach and move predictions to cpu since will be processing using numpy arrays which are CPU only
    conf_pred_cpu = conf_pred.detach().cpu()
    loc_pred_cpu  = loc_pred.detach().cpu()

    # FloatTensor = torch.cuda.FloatTensor if conf_pred.is_cuda else torch.FloatTensor
    FloatTensor = torch.FloatTensor

    # Create a zero tensor that is the same size as the pred tensor (should be the size of the image)
    target_states = torch.zeros(conf_pred_cpu.shape[0:3],requires_grad=False).type(FloatTensor) # BxHxW
    target_locations_rsz = torch.zeros(loc_pred_cpu.shape,requires_grad=False).type(FloatTensor) # BxHxWx2

    for b in range(conf_pred.shape[0]):

        # Get indices for pixels that have an object in them and set value to 1
        # x locations are columns (index 1)! y locations are rows (index 0)!
        target_b_idx = target_locations[b].detach().cpu().floor().int().numpy().transpose()
        target_states[b,target_b_idx[1],target_b_idx[0]] = 1

        # Add the normalized location (absolute location - which pixel it is) to tensor for MSELoss
        x_gt = target_locations[b].detach().cpu().transpose(0,1)[0] - target_b_idx[0].astype(np.float32)
        y_gt = target_locations[b].detach().cpu().transpose(0,1)[1] - target_b_idx[1].astype(np.float32)


        target_locations_rsz[b,target_b_idx[1],target_b_idx[0],0] = x_gt 
        target_locations_rsz[b,target_b_idx[1],target_b_idx[0],1] = y_gt 

    return target_states, target_locations_rsz

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.alpha   = torch.tensor(0.25)
        self.gamma   = 2.0
        self.pos_weight = torch.tensor(25)

    def forward(self, classification, target_states,device, mode):
        """ Args:
                classification (torch.Tensor): [B, sum(AHW)] logits
                anchor_states  (torch.Tensor): [B, sum(AHW)]
        """
        p = torch.sigmoid(classification) # Head returns logits

        positive_indices = torch.eq(target_states,  1)
        ignore_indices   = torch.eq(target_states, -1)

        num_positive_indices = positive_indices.sum()

        # Focal loss
        # - Apply alpha to anchors with IoU > 0.5, 1-alpha to IoU < 0.5
        alpha = torch.where(positive_indices, self.alpha.to(device), 1 - self.alpha.to(device))
        focal_weight = torch.where(positive_indices, 1 - p, p)
        focal_weight = alpha * focal_weight.pow(self.gamma)

        cls_loss = focal_weight * F.binary_cross_entropy_with_logits(
                    classification, target_states, reduction='none',pos_weight=self.pos_weight.to(device))

        # Ignore (zero loss) 0.4 < iou < 0.5
        zeros    = torch.zeros_like(cls_loss)
        cls_loss = torch.where(ignore_indices, zeros, cls_loss)

        if mode == 'train':
            cls_loss = cls_loss.sum().div(torch.clamp(num_positive_indices, min=1.0))
        else:
            cls_loss = cls_loss.sum()

        return cls_loss

class MSELoss_Custom(nn.Module):
    def __init__(self,reduction):
        super(MSELoss_Custom,self).__init__()
        self.loss = torch.nn.MSELoss(reduction=reduction)
    def forward(self, pred, target_locations, target_states):

        '''
        target_locations should be the same shape as the prediction tensor (B,W,H,2) where the 2 refers to (x,y) ground truth locations!
        '''
        
        x_gt = target_locations[...,0]
        y_gt = target_locations[...,1]

        # Pass x,y locations through sigmoid to normalize between 0 and 1 for each pixel
        x = torch.sigmoid(pred[...,0])
        y = torch.sigmoid(pred[...,1])

        # We only want to calculate the MSE loss for the predictions that have a ground truth in that pixel
        target_states_bool = target_states>0

        # Filter the tensors to just pixels that have a ground truth
        x_gt_filtered = x_gt[target_states_bool]
        y_gt_filtered = y_gt[target_states_bool]
        x_filtered    = x[target_states_bool]
        y_filtered    = y[target_states_bool]

        # Stack tensors to make one filtered gt tensor and one filtered pred tensor
        predxy = torch.stack((x_filtered,y_filtered),dim=1)
        gtxy   = torch.stack((x_gt_filtered,y_gt_filtered),dim=1)

        # Calculate MSE Loss
        reg_loss = self.loss(predxy,gtxy)

        return reg_loss

"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 10/02/2019 
"""
