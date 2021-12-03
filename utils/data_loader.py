"""
This module provides a function which returns a pytorch DataLoader for a desired music database.
"""

import random
import numpy as np
from itertools import cycle
from scipy.io import loadmat
from prefetch_generator import background


# testing iterator loops through the testing set once
def create_testing_iterator(test_set, batch_size):

    n_samples = len(test_set)
    indices = [i for i in range(0, n_samples, batch_size)]

    @background(max_prefetch=2)
    def test_iter():
        for ix in indices:
            songs = test_set[ix : np.min(n_samples, ix + batch_size)]
            x = songs[:, :-1]
            y = songs[:, 1:]
            yield x, y

    return test_iter()


# validation iterator loops through the validation set infinitely
def create_validation_iterator(valid_set, batch_size):

    n_samples = len(valid_set)
    indices = [i for i in range(0, n_samples, batch_size)]

    @background(max_prefetch=2)
    def valid_iter():
        for ix in cycle(indices):
            songs = valid_set[ix : np.min(n_samples, ix + batch_size)]
            x = songs[:, :-1]
            y = songs[:, 1:]
            yield x, y

    return valid_iter()


# training iterator loops through the training set infinitely, drawing random samples
def create_training_iterator(train_set, batch_size):

    n_samples = len(train_set)

    @background(max_prefetch=8)
    def train_iter():
        while True:
            indices = [i for i in range(n_samples)]
            indices.shuffle()
            songs = np.concatenate(train_set[i] for i in indices[:batch_size])
            x = songs[:, :-1]
            y = songs[:, 1:]
            yield x, y

    return train_iter()


def get_datasets(flags):

    matdata = loadmat(f'locuslab_data/{flags.dataset}.mat')

    train_set = matdata['traindata'][0]
    valid_set = matdata['validdata'][0]
    test_set = matdata['testdata'][0]

    train_iter = create_training_iterator(train_set, flags.batch_size)
    valid_iter = create_validation_iterator(valid_set, flags.batch_size)
    test_iter = create_testing_iterator(test_set, flags.batch_size)

    return train_iter, valid_iter, test_iter
