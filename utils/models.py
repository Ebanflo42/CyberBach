"""
This module handles the readout we put on top of models.
"""

import numpy as np
import numpy.linalg as la

import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions.distribution import Distribution

from math import sin, cos


def get_constructor(architecture: str):
    """
    :param architecture: name of the architecture we want to use
    :param cuda: whether or not the model needs to be run on the gpu
    """

    if architecture == 'GRU':
        return nn.GRU
    elif architecture == 'LSTM':
        return nn.LSTM
    elif architecture == 'TANH':
        return nn.RNN
    else:
        raise ValueError("Architecture {} not recognized.".format(architecture))


# A pytorch model together with a linear read-out
class FullRNN(nn.Module):

    def __init__(self, flags, architecture: str, n_rec: int):

        super(FullRNN, self).__init__()

        self.architecture = architecture

        # construct linear readout
        self.n_in = 88
        self.n_rec = n_rec
        self.n_out = 88
        self.output_weights = nn.Linear(self.n_rec, self.n_out)

        # get constructor for input and hidden layers
        constructor = get_constructor(self.architecture)

        # construct the model based on whether it is implemented in pytorch or base_models.py
        self.rnn = constructor(input_size=self.n_in, hidden_size=self.n_rec, num_layers=1, batch_first=True)

        # gradient clipping if we want it
        if flags.use_grad_clip:
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -flags.grad_clip, flags.grad_clip))

    def forward(self, x):

        hiddens, hn = self.rnn(x)
        output = self.output_weights(hiddens)
        return output, hn

