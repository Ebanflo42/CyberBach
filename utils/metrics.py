"""
Here we define pytorch modules for efficiently computing accuracy and loss.
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

    def forward(self, output, target):

        N = output.shape[0]

        prediction = (torch.sigmoid(output) > 0.5).type(torch.float32)

        # sum over notes
        tru_pos = torch.sum(prediction*target, dim=2)
        # Bay et al sum over time but this yields way higher results than Boulanger-Lewandowski
        #tru_pos = torch.sum(tru_pos, dim=1)

        # compute accuracy for all sequences at each time point
        T = output.shape[1]
        acc_over_time = []

        for t in range(T):

            # get false positives and negatives for each sequence
            false_pos = torch.sum(prediction[:, t]*(1 - target[:, t]), dim=1)
            false_neg = torch.sum((1 - prediction[:, t])*target[:, t], dim=1)

            # quick trick to try to avoid NaNs
            false_pos += F.relu(1 - tru_pos[:, t])

            # true negatives are unremarkable for sparse binary sequences
            this_acc = tru_pos[:, t]/(tru_pos[:, t] + false_pos + false_neg)

            acc_over_time.append(this_acc)

        # first take the average for each sequence, then sum over sequences
        result = torch.cat(acc_over_time).reshape(T, N)
        result = torch.mean(result)

        return result
