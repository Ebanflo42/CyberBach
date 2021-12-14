"""
I use `prefetch_generator`s because they are quick and easy to construct and are inter-operable with PyTorch, TensorFlow, SciKit-learn, or whatever you like.

This script provides utilities for going from a dataset name to iterators for the training, validations, and testing data.
"""

import torch
import random
import numpy as np

from itertools import cycle
from scipy.io import loadmat
from prefetch_generator import background


# we want every sequence in a batch to have the same length
# so we first find the length of the longset sequence
def get_max_seq_len(train_set, valid_set, test_set):

    seqlen = 0
    for s in train_set:
        if len(s) > seqlen:
            seqlen = len(s)
    for s in valid_set:
        if len(s) > seqlen:
            seqlen = len(s)
    for s in test_set:
        if len(s) > seqlen:
            seqlen = len(s)

    return seqlen


# pad sequence with zeros and return the corresponding mask
def pad_seq(seq, seqlen):
    result = np.concatenate((seq, np.zeros((seqlen - len(seq), 88), dtype=seq.dtype)), axis=0)
    mask = np.concatenate((np.ones(len(seq) - 1, dtype=np.float32), np.zeros(seqlen - len(seq), dtype=np.float32)))
    return result, mask


# testing iterator loops through the testing set once
def create_testing_iterator(test_set, batch_size, seqlen):

    n_samples = len(test_set)
    indices = [i for i in range(0, n_samples, batch_size)]

    @background(max_prefetch=2)
    def test_iter():
        for ix in indices:
            padded_songs = [pad_seq(test_set[i], seqlen) for i in range(
                ix, np.minimum(n_samples, ix + batch_size))]
            songs, masks = np.stack([s[0] for s in padded_songs]), np.stack([s[1] for s in padded_songs])
            x = torch.tensor(songs[:, :-1], dtype=torch.float32)
            y = torch.tensor(songs[:, 1:], dtype=torch.float32)
            mask = torch.tensor(masks, dtype=torch.float32)
            yield x, y, mask

    return test_iter()


# validation iterator loops through the validation set infinitely
def create_validation_iterator(valid_set, batch_size, seqlen):

    n_samples = len(valid_set)
    indices = [i for i in range(0, n_samples, batch_size)]

    @background(max_prefetch=2)
    def valid_iter():
        for ix in cycle(indices):
            padded_songs = [pad_seq(valid_set[i], seqlen) for i in range(
                ix, np.minimum(n_samples, ix + batch_size))]
            songs, masks = np.stack([s[0] for s in padded_songs]), np.stack([s[1] for s in padded_songs])
            x = torch.tensor(songs[:, :-1], dtype=torch.float32)
            y = torch.tensor(songs[:, 1:], dtype=torch.float32)
            mask = torch.tensor(masks, dtype=torch.float32)
            yield x, y, mask

    return valid_iter()


# training iterator loops through the training set infinitely, drawing random samples
def create_training_iterator(train_set, batch_size, seqlen):

    n_samples = len(train_set)

    @background(max_prefetch=8)
    def train_iter():
        while True:
            indices = [i for i in range(n_samples)]
            random.shuffle(indices)
            padded_songs = [pad_seq(train_set[i], seqlen) for i in indices[:batch_size]]
            songs, masks = np.stack([s[0] for s in padded_songs]), np.stack([s[1] for s in padded_songs])
            x = torch.tensor(songs[:, :-1], dtype=torch.float32)
            y = torch.tensor(songs[:, 1:], dtype=torch.float32)
            mask = torch.tensor(masks, dtype=torch.float32)
            yield x, y, mask

    return train_iter()


def get_datasets(FLAGS):

    matdata = loadmat(f'locuslab_data/{FLAGS.dataset}.mat')

    train_set = matdata['traindata'][0]
    valid_set = matdata['validdata'][0]
    test_set = matdata['testdata'][0]

    seqlen = get_max_seq_len(train_set, valid_set, test_set)

    train_iter = create_training_iterator(train_set, FLAGS.batch_size, seqlen)
    valid_iter = create_validation_iterator(valid_set, FLAGS.batch_size, seqlen)
    test_iter = create_testing_iterator(test_set, FLAGS.batch_size, seqlen)

    return train_iter, valid_iter, test_iter
