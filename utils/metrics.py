"""
The two PyTorch modules here define custom loss functions for training the model and evaluating performance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# see Bay et al 2009 for the definition of frame-level accuracy
# this module also returns the mean over all sequences
class FrameAccuracy(nn.Module):

    def __init__(self):

        super(FrameAccuracy, self).__init__()

    def forward(self, output, target, mask):

        N = output.shape[0]

        prediction = (output > 0.0).type(torch.float32)

        # sum over notes
        tru_pos = torch.sum(prediction*target, dim=2)

        # get false positives and negatives for each sequence
        false_pos = torch.sum(prediction*(1 - target), dim=2)
        false_neg = torch.sum((1 - prediction)*target, dim=2)

        # true negatives are unremarkable for sparse binary sequences
        # this gives accuracy at each batch and time step
        acc = torch.nan_to_num(tru_pos/(tru_pos + false_pos + false_neg), nan=1.0)

        # apply the mask by taking the mean only over the parts of the sequences which matter
        masked_acc = torch.sum(mask*acc)/torch.sum(mask)

        return masked_acc


# average binary cross entropy per time step (from logits)
# with a mask to indicate where the data actually is (songs in the same batch have different length)
class MaskedBCE(nn.Module):

    def __init__(self):

        super(MaskedBCE, self).__init__()

    def forward(self, output, target, mask):
        '''
        :param output: time-dependent output of a recurrent neural network (logits)
        :param target: binary sequence (should be float32) in indicating which notes should be played
        :param mask: binary sequence (also float32) with ones where the actual song is and 0s elsewhere, should not include a dimensionfor notes
        :return: binary cross entropy averaged over every time step of every song in the batch
        '''

        surprisal = F.log_softmax(output, dim=-1)

        # taking the sum and dividing by the sum of the mask gaurantees that
        # only the relevant time steps will contribute to the loss
        # i.e. this ends up as an average over batch and time
        mask_factor = torch.tile(torch.reshape(
            mask, (mask.shape[0], mask.shape[1], 1)), (1, 1, target.shape[2]))
        ce = -torch.sum(mask_factor*target*surprisal)
        result = ce/torch.sum(mask)

        return result
