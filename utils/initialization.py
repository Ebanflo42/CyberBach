"""
This module handles the initialization of models.

If readout=None, initialization will always be default
Ortherwise, it will depend on the architecture and whether it is implemented by me or pytorch.

The way initialization works is that all models will be initialized based on a simpler, pre-trained model, with some tweaks specified by the initializer dictionary. The 'path' entry of the initializer dictionary tells us where the state dictionary of the pre-trained model is.
"""

import numpy as np
import numpy.linalg as la

import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import Bernoulli, Uniform


def make_limit_cycle_weights(shape: torch.Size):

    if shape[0] != shape[1] or shape[0]%2 == 1:
        raise ValueError("Tried to get block orthogonal matrix of shape {}".format(str(shape)))

    n = shape[0]//2
    result = torch.zeros(shape)

    bern = Bernoulli(torch.tensor([0.5]))
    uni = Uniform(0.2*np.pi, 0.4*np.pi)

    for i in range(n):

        t = uni.sample()
        if bern.sample == 1:
            t = -t
        scale = 1 + 0.2*uni.sample()

        mat = scale*torch.tensor([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

        result[2*i : 2*(i + 1), 2*i : 2*(i + 1)] = mat

    return result


def initialize(model, flags):

    if flags.initialization == 'default':
        pass

    elif flags.initialization == 'orthogonal':
        for param in model.rnn.parameters():
            if len(param.shape) > 1:
                init.orthogonal_(param)

    elif flags.initialization == 'limit_cycle':

        if model.architecture == 'TANH':
            sd = model.state_dict()
            sd['rnn.weight_hh_l0'] = make_limit_cycle_weights(sd['rnn.weight_hh_l0'].shape)
            sd['rnn.bias_hh_l0'] = torch.zeros_like(sd['rnn.bias_hh_l0'])
            sd['rnn.bias_ih_l0'] = torch.zeros_like(sd['rnn.bias_ih_l0'])
            model.load_state_dict(sd)

        elif model.architecture == 'GRU':
            sd = model.state_dict()
            hh = sd['rnn.weight_hh_l0']
            lc_weights = make_limit_cycle_weights((hh.shape[1], hh.shape[1]))
            sd['rnn.weight_hh_l0'] = 2*torch.cat([torch.zeros_like(hh[:2*hh.shape[1]]), lc_weights])
            sd['rnn.bias_hh_l0'] = torch.zeros_like(sd['rnn.bias_hh_l0'])
            sd['rnn.bias_ih_l0'] = torch.zeros_like(sd['rnn.bias_ih_l0'])
            model.load_state_dict(sd)

        else:
            raise ValueError(f'Architecture {model.architecture} does not support limit cycle initialiation.')

    else:
        raise ValueError(f'Initialization {flags.initialization} not recognized.')