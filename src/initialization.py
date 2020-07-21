"""
This module handles the initialization of models.

If readout=None, initialization will always be default
Ortherwise, it will depend on the architecture and whether it is implemented by me or pytorch.

The way initialization works is that all models will be initialized based on a simpler, pre-trained model, with some tweaks specified by the initializer dictionary. The 'path' entry of the initializer dictionary tells us where the state dictionary of the pre-trained model is.
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


def make_identity(shape: torch.Size) -> torch.Tensor:
    """
    :param shape: shape of the desired matrix
    :return: matrix which is zeroes everywhere and ones on diagonal
    """

    result = torch.zeros(shape)

    for i in range(np.minimum(shape[0], shape[1])):
        result[i, i] = 1.0

    return result


def make_block_ortho(shape: torch.Size,
                     t_distrib: Distribution,
                     architecture="",
                     parity=None) -> torch.Tensor:
    """
    :param shape: shape of the block orthogonal matrix that we desire
    :param t_distrib: pytorch distribution from which we will sample the angles
    :param parity: None will return a mixed parity block orthogonal matrix, 'reflect' will return a decoupled system of 2D reflections, and anything else will return decoupled system of rotations.
    """

    if shape[0] != shape[1] or shape[0]%2 == 1:
        raise ValueError("Tried to get block orthogonal matrix of shape {}".format(str(shape)))

    n = shape[0]//2
    result = torch.zeros(shape)

    bern = Bernoulli(torch.tensor([0.5]))

    for i in range(n):

        t = t_distrib.sample()
        mat = torch.tensor([[cos(t), sin(t)], [-sin(t), cos(t)]])

        if parity == None:
            mat[0] *= 2*bern.sample() - 1
        elif parity == 'reflect':
            mat[0] *= -1

        result[2*i : 2*(i + 1), 2*i : 2*(i + 1)] = mat

    return result


def _initialize_lds(model: nn.Module, initializer: dict) -> nn.Module:
    """
    :param model: latent linear dynamical system
    :param initializer: initialize the hidden weights based on this dictionary
    initialize the model in place based on the weights of a trained regression model
    """

    reg_sd = torch.load(initializer['path'])

    # top block of input weights will be identity, everything else will be random
    sq = torch.Size([88, 88])
    model.rnn.weight_ih_l0.weight.data[0 : 88, 0 : 88] = make_identity(sq)

    # output weights are based on regression weights
    model.output_weights.weight.data = torch.zeros(model.output_weights.weight.data.shape)
    model.output_weights.weight.data[0 : 88, 0 : 88] = reg_sd['weights.weight']

    # the hidden matrix must erase the systems memory at each time step
    hid_size = model.rnn.weight_hh_l0.weight.data.shape
    model.rnn.weight_hh_l0.weight.data = torch.zeros(hid_size)

    slen = hid_size[0] - 88
    sub_size = torch.Size([slen, slen])
    scale = initializer['scale']

    if initializer['init'] == 'blockortho':
        t_distrib = initializer['t_distrib']
        parity = initializer['parity']
        model.rnn.weight_hh_l0.weight.data[88:, 88:] = scale*make_block_ortho(sub_size, t_distrib, parity=parity)

    elif initializer['init'] == 'identity':
        model.rnn.weight_hh_l0.weight.data[88:, 88:] = scale*make_identity(sub_size)

    elif initializer['init'] == 'zero':
        pass

    elif initializer['init'] == 'normal':
        model.rnn.weight_hh_l0.weight.data[88:, 88:] = scale*torch.randn(sub_size)

    elif initializer['init'] == "critical":

        reg_sd = torch.load(initializer['path'])

        sq = torch.Size([88, 88])
        model.rnn.weight_ih_l0.data[0 : 88, 0 : 88] = initializer['scale']*make_identity(sq)

        hid_shape = model.rnn.weight_hh_l0.data.shape
        model.rnn.weight_hh_l0.data = torch.zeros(hid_shape)
        for i in range(88, hid_shape[0]):
            model.rnn.weight_hh_l0.data[i, i] = 1.0/(i - 87)
            if i < hid_shape[0] - 1:
                model.rnn.weight_hh_l0.data[i, i + 1] = 1

        out_shape = model.output_weights.weight.data.shape
        model.output_weights.weight.data = torch.zeros(out_shape)
        model.output_weights.weight.data[0 : 88, 0 : 88] = reg_sd['weights.weight']

    elif initializer['init'] != 'default':
        raise ValueError("Initialization {} not recognized.".format(initializer['init']))


def _initialize_tanh(model: nn.Module, initializer: dict) -> nn.Module:
    """
    :param model: tanh recurrent neural network
    :param initializer: initialize the hidden weights based on this dictionary
    Initialize the model in place based on the weights of a trained linear dynamical system. Since the architecture is so similar to LDS, there are no tweaks applied.
    """

    if initializer['init'] == "lds":

        lds_sd = torch.load(initializer['path'])

        model.rnn.weight_ih_l0.data = lds_sd['rnn.weight_ih_l0.weight']
        model.rnn.weight_hh_l0.data = lds_sd['rnn.weight_hh_l0.weight']
        model.output_weights.weight.data = lds_sd['output_weights.weight']

    elif initializer['init'] == "critical":

        reg_sd = torch.load(initializer['path'])

        sq = torch.Size([88, 88])
        model.rnn.weight_ih_l0.data[0 : 88, 0 : 88] = initializer['scale']*make_identity(sq)

        hid_shape = model.rnn.weight_hh_l0.data.shape
        model.rnn.weight_hh_l0.data = torch.zeros(hid_shape)
        for i in range(88, hid_shape[0]):
            model.rnn.weight_hh_l0.data[i, i] = 1.0/(i - 87)
            if i < hid_shape[0] - 1:
                model.rnn.weight_hh_l0.data[i, i + 1] = 1

        out_shape = model.output_weights.weight.data.shape
        model.output_weights.weight.data = torch.zeros(out_shape)
        model.output_weights.weight.data[0 : 88, 0 : 88] = reg_sd['weights.weight']

    elif initializer['init'] != 'default':
        raise ValueError("Initialization {} not recognized.".format(initializer['init']))


def _initialize(model: nn.Module, initializer: dict, architecture: str) -> nn.Module:
    """
    :param model: model to be initialized
    :param initializer: a dictionary specifying all information about the desired initialization
    :param architecture: we must know the architecture so that we know the structure of the parameters
    Initialize the model in-place.
    """

    if architecture == "LDS":
        _initialize_lds(model, initializer)

    elif architecture == "TANH":
        _initialize_tanh(model, initializer)
