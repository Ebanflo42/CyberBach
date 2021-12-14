"""
This module handles the initialization of models.

Initialization may be `default` (Xavier), `orthogonal`, or `limit_cycle` (only for TANH and GRU).
"""

import numpy as np
import numpy.linalg as la

import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import Bernoulli, Uniform

from utils.models import MusicRNN


def make_limit_cycle_weights(shape: torch.Size):
    '''
    :param shape: Desired shape of the weight matrix, must be square and have dimensions divisible by two.
    :return: Recurrent weight matrix for TANH recurrent neural network which will give the network the dynamics of a direct sum of limit cycles.
    '''

    if shape[0] != shape[1] or shape[0]%2 == 1:
        raise ValueError("Tried to get block orthogonal matrix of shape {}".format(str(shape)))

    n = shape[0]//2
    result = torch.zeros(shape)

    bern = Bernoulli(torch.tensor([0.5]))
    uni = Uniform(0.2*np.pi, 0.4*np.pi)

    for i in range(n):

        # rotation angle
        t = uni.sample()
        if bern.sample == 1:
            t = -t

        # scale factor
        # at 1, the initialization will be an instance of orthogonal, which has a stable fixed point
        # by scaling up a bit we come close to bifurcating into a limit cycle
        scale = 1 + 0.2*uni.sample()

        mat = scale*torch.tensor([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

        result[2*i : 2*(i + 1), 2*i : 2*(i + 1)] = mat

    return result


def initialize(model: MusicRNN, FLAGS):
    '''
    :param model: Model to be initialized
    :param FLAGS: Flags for the experiment. Determines which initialization to use.
    Initialize the given model to the desired initialization.
    '''

    if FLAGS.initialization == 'default':
        pass

    elif FLAGS.initialization == 'orthogonal':
        for param in model.rnn.parameters():
            if len(param.shape) > 1:
                init.orthogonal_(param)

    elif FLAGS.initialization == 'limit_cycle':

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
        raise ValueError(f'Initialization {FLAGS.initialization} not recognized.')