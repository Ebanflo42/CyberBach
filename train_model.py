import os
import json
import string
import random
import datetime
import pickle as pkl

import absl
import simmanager
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.models import FullRNN
from utils.data_loader import get_loader
from utils.metrics import MaskedBCE, compute_acc, compute_loss

from torch.utils.data import DataLoader
from torch import Tensor, device
from copy import deepcopy
from time import sleep
from tqdm import tqdm
from itertools import product
from os.path import join as opj


# this context is used when we are running things on the cpu
class NullContext(object):
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass


# a single optimization step
def train_iter(sm: simmanager.SimManager,
               device: device,
               input_tensor: Tensor,
               target: Tensor,
               mask: Tensor,
               model: nn.Module,
               loss_fcn: nn.Module,
               optimizer: optim.Optimizer):

    input_tensor = input_tensor.to(device)

    output, hidden_tensors = model(input_tensor)

    loss = loss_fcn(output, target, mask, model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_loop(flags, model):

    # if we are on cuda we construct the device and run everything on it
    cuda_device = NullContext()
    device = torch.device('cpu')
    if flags.use_gpu:
        dev_name = 'cuda:' + str(gpu)
        cuda_device = torch.cuda.device(dev_name)
        device = torch.device(dev_name)
        model = model.to(device)

    with cuda_device:

        # see metrics.py
        loss_fcn = MaskedBCE(
            regularization, low_off_notes=low_off_notes, high_off_notes=high_off_notes)

        # compute the metrics before training and log them
        if logging:

            train_loss = compute_loss(loss_fcn, model, train_loader)
            test_loss = compute_loss(loss_fcn, model, test_loader)
            val_loss = compute_loss(loss_fcn, model, valid_loader)

            _run.log_scalar("trainLoss", train_loss)
            _run.log_scalar("testLoss", test_loss)
            _run.log_scalar("validLoss", val_loss)

            train_acc = compute_acc(
                model, train_loader, low=low_off_notes, high=high_off_notes)
            test_acc = compute_acc(
                model, test_loader, low=low_off_notes, high=high_off_notes)
            val_acc = compute_acc(model, valid_loader,
                                  low=low_off_notes, high=high_off_notes)

            _run.log_scalar("trainAccuracy", train_acc)
            _run.log_scalar("testAccuracy", test_acc)
            _run.log_scalar("validAccuracy", val_acc)

        # construct the optimizer
        optimizer = None
        if optmzr == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optmzr == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optmzr == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError("Optimizer {} not recognized.".format(optmzr))

        # learning rate decay
        scheduler = None
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: decay**epoch)

        # begin training loop
        for epoch in tqdm(range(num_epochs)):

            for input_tensor, target, mask in train_loader:
                train_iter(device,
                           cuda_device,
                           input_tensor,
                           target,
                           mask,
                           model,
                           loss_fcn,
                           optimizer,
                           save_every_epoch,
                           save_dir,
                           train_loader,
                           test_loader,
                           valid_loader,
                           low_off_notes,
                           high_off_notes,
                           _log,
                           _run,
                           logging=logging)

            # learning rate decay
            scheduler.step()

            # use sacred to log testing and validation loss and accuracy
            if logging:

                test_loss = compute_loss(loss_fcn, model, test_loader)
                val_loss = compute_loss(loss_fcn, model, valid_loader)
                test_acc = compute_acc(
                    model, test_loader, low=low_off_notes, high=high_off_notes)
                val_acc = compute_acc(
                    model, valid_loader, low=low_off_notes, high=high_off_notes)

                _run.log_scalar("testLoss", test_loss)
                _run.log_scalar("validLoss", val_loss)
                _run.log_scalar("testAccuracy", test_acc)
                _run.log_scalar("validAccuracy", val_acc)

        # save a copy of the trained model and make sacred remember it
        if save_final_model and logging:
            fin_sd = deepcopy(model.state_dict())
            torch.save(fin_sd, save_dir + 'final_state_dict.pt')
            _run.add_artifact(save_dir + 'final_state_dict.pt')

    # recompute the metrics so that this function can return them
    train_loss = compute_loss(loss_fcn, model, train_loader)
    test_loss = compute_loss(loss_fcn, model, test_loader)
    val_loss = compute_loss(loss_fcn, model, valid_loader)

    train_acc = compute_acc(model, train_loader,
                            low=low_off_notes, high=high_off_notes)
    test_acc = compute_acc(
        model, test_loader, low=low_off_notes, high=high_off_notes)
    val_acc = compute_acc(model, valid_loader,
                          low=low_off_notes, high=high_off_notes)

    return ((train_loss, test_loss, val_loss), (train_acc, test_acc, val_acc))


def main(_argv):

    flags = absl.flags.FLAGS

    # construct simulation manager
    base_path = opj(flags.results_dir, flags.exp_name)
    identifier = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    sim_name = 'run_{}'.format(identifier)
    sm = simmanager.SimManager(sim_name, base_path, write_protect_dirs=False, tee_sdx_to='output.log')

    with sm:

        # check for cuda
        if flags.use_gpu and not torch.cuda.is_available():
            raise OSError('CUDA is not available. Check your installation or set `use_gpu` to False.')

        # dump flags for this experiment
        with open(opj(sm.data_path, 'flags.json'), 'w') as f:
            json.dump(flags, f)

        # generate, save, and set random seed
        random_seed = datetime.now().microsecond
        np.save(opj(sm.data_path, 'random_seed'), random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # check for old model to restore from
        old_model_dict = None
        architecture = flags.architecture
        n_rec = flags.n_rec
        if flags.restore_from != '':
            with open(opj(flags.restore_from, 'results', 'model_checkpoint.pkl'), 'rb') as f:
                old_model_dict = pkl.load(f)
            with open(opj(flags.restore_from, 'data', 'flags.json'), 'r') as f:
                old_flags = json.load(f.read())
                architecture = old_flags['architecture']
                n_rec = old_flags['n_rec']

        model = FullRNN(flags, architecture, n_rec)

if __name__ == '__main__':

    # system
    absl.flags.DEFINE_bool(
        'use_gpu', False, 'Whether or not to use the GPU. Fails if True and CUDA is not available.')
    absl.flags.DEFINE_string(
        'exp_name', '', 'If non-empty works as a special directory for the experiment')
    absl.flags.DEFINE_string('result_dir', 'trained_models',
                             'Name of the directory to save all results within.')

    # training
    absl.flags.DEFINE_enum('dataset', 'JSB_Chorales', [
                           'JSB_Chorales', 'Nottingham', 'Piano_midi', 'MuseData'], 'Which dataset to train the model on.')
    absl.flags.DEFINE_integer('train_iters', 20000,
                              'How many training batches to show the network.')
    absl.flags.DEFINE_integer('batch_size', 100, 'Batch size.')
    absl.flags.DEFINE_float('lr', 0.001, 'Learning rate.')
    absl.flags.DEFINE_integer(
        'decay_every', 100, 'Shrink the learning rate after this many batches.')
    absl.flags.DEFINE_float(
        'lr_decay', 0.95, 'Shrink the learning rate by this factor.')
    absl.flags.DEFINE_enum('optimizer', 'Adam', [
                           'Adam', 'SGD', 'Adagrad', 'RMSprop'], 'Which optimizer to use.')
    absl.flags.DEFINE_float('reg_coeff', 0.0001,
                            'Coefficient for L2 regularization of weights.')
    absl.flags.DEFINE_bool(
        'use_grad_clip', False, 'Whether or not to clip the backward gradients by their magnitude.')
    absl.flags.DEFINE_float(
        'grad_clip', 1, 'Maximum magnitude of gradients if gradient clipping is used.')
    absl.flags.DEFINE_integer('validate_every', 100, 'Validate the model at this many training steps.')
    absl.flags.DEFINE_integer('save_every', 200, 'Save the model at this many training steps.')

    # model
    absl.flags.DEFINE_enum('architecture', 'TANH', [
                           'TANH', 'LSTM', 'GRU'], 'Which recurrent architecture to use.')
    absl.flags.DEFINE_integer(
        'n_rec', 1024, 'How many recurrent neurons to use.')
    absl.flags.DEFINE_enum('initialization', 'default', ['default', 'orthogonal', 'limit_cycle'],
                           'Which initialization to use for the recurrent weight matrices. Default is uniformly distributed weights. Limit cycles only apply to TANH and GRU')
    absl.flags.DEFINE_string(
        'restore_from', '', 'If non-empty, restore all the previous model from this directory and train it using the new flags.')


    absl.app.run(main)

