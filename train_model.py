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

from utils.models import MusicRNN
from utils.data_loader import get_datasets
from utils.metrics import FrameAccuracy

from torch import Tensor, device
from torch.nn import BCELoss
from copy import deepcopy
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


def train_loop(sm, flags, model, train_iter, valid_iter, test_iter):

    # if we are on cuda we construct the device and run everything on it
    cuda_device = NullContext()
    device = torch.device('cpu')
    if flags.use_gpu:
        dev_name = 'cuda:0'
        cuda_device = torch.cuda.device(dev_name)
        device = torch.device(dev_name)
        model = model.to(device)

    train_loss = []
    train_reg = []
    train_acc = []

    valid_loss = []
    valid_acc = []

    with cuda_device:

        # construct the optimizer
        if flags.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=flags.lr)
        elif flags.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=flags.lr)
        elif flags.optimizer == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=flags.lr)
        elif flags.optimizer == "Adagrad":
            optimizer = optim.Adagrad(model.parameters(), lr=flags.lr)
        else:
            raise ValueError(
                "Optimizer {} not recognized.".format(flags.optimizer))

        # learning rate decay
        scheduler = None
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: flags.lr_decay**(epoch//flags.decay_every))

        acc_fcn = FrameAccuracy()
        loss_fcn = BCELoss()

        # begin training loop
        for i in range(flags.train_iters):

            # get next training sample
            x, y = next(train_iter)
            x, y = torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(
                y, dtype=torch.float32, device=device)

            # forward pass
            output, hidden = model(x)

            # binary cross entropy
            bce_loss = loss_fcn(output, y)

            # weight regularization
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += flags.reg_coeff*torch.norm(param)

            # backward pass and optimization step
            loss = bce_loss + l2_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # learning rate decay
            scheduler.step()

            # compute accuracy
            acc = acc_fcn(output, y)

            # append metrics
            train_loss.append(bce_loss)
            train_acc.append(acc)
            train_reg.append(l2_reg)

            if i > 0 and i % flags.validate_every == 0:

                print(
                    f'Validating at iteration {i}.\n  Training loss: {bce_loss}\n  Training accuracy: {acc}\n  L2 regularization: {l2_reg}')

                # get next validation sample
                x, y = next(valid_iter)
                x, y = torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(
                    y, dtype=torch.float32, device=device)

                # forward pass
                output, hidden = model(x)

                # binary cross entropy
                bce_loss = loss_fcn(output, y)

                # compute accuracy
                acc = acc_fcn(output, y)

                # append metrics
                valid_loss.append(bce_loss)
                valid_acc.append(acc)

                print(
                    f'  Validation loss: {bce_loss}\n  Validation accuracy: {acc}')

            if i > 0 and i % flags.save_every == 0:

                print(f'Saving at iteration {i}.')

                np.save(opj(sm.results_path, 'training_loss'), train_loss)
                np.save(opj(sm.results_path, 'training_accuracy'), train_acc)
                np.save(opj(sm.results_path, 'training_regularization'), train_reg)

                np.save(opj(sm.results_path, 'validation_loss'), valid_loss)
                np.save(opj(sm.results_path, 'validation_accuracy'), valid_acc)

                torch.save(model.state_dict(), opj(
                    sm.results_path, 'model_checkpoint'))

        print('Finished training. Entering testing phase.')

        test_loss = []
        test_acc = []
        tot_test_samples = 0

        for x, y in test_iter:

            x, y = torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(
                y, dtype=torch.float32, device=device)

            output, hidden = model(x)

            bce_loss = loss_fcn(output, y)
            acc = acc_fcn(output, y)

            batch_size = x.shape[0]
            tot_test_samples += batch_size
            test_loss.append(batch_size*bce_loss)
            test_acc.append(batch_size*acc)

        final_test_loss = np.sum(test_loss)/tot_test_samples
        final_test_acc = np.sum(test_acc)/tot_test_samples
        print(
            f'  Testing loss: {final_test_loss}\n  Testing accuracy: {final_test_acc}')

        print('Final save.')
        np.save(opj(sm.results_path, 'testing_loss'), final_test_loss)
        np.save(opj(sm.results_path, 'testing_accuracy'), final_test_acc)
        torch.save(model.state_dict(), opj(
            sm.results_path, 'model_checkpoint'))


def main(_argv):

    flags = absl.flags.FLAGS

    # construct simulation manager
    base_path = opj(flags.results_dir, flags.exp_name)
    identifier = ''.join(random.choice(
        string.ascii_lowercase + string.digits) for _ in range(4))
    sim_name = 'run_{}'.format(identifier)
    sm = simmanager.SimManager(
        sim_name, base_path, write_protect_dirs=False, tee_sdx_to='output.log')

    with sm:

        # check for cuda
        if flags.use_gpu and not torch.cuda.is_available():
            raise OSError(
                'CUDA is not available. Check your installation or set `use_gpu` to False.')

        # dump flags for this experiment
        with open(opj(sm.data_path, 'flags.json'), 'w') as f:
            json.dump(flags, f)

        # generate, save, and set random seed
        if flags.random_seed != -1:
            random_seed = flags.random_seed
        else:
            random_seed = datetime.now().microsecond
        np.save(opj(sm.data_path, 'random_seed'), random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # check for old model to restore from
        if flags.restore_from != '':

            with open(opj(flags.restore_from, 'data', 'flags.json'), 'r') as f:
                old_flags = json.load(f.read())

            architecture = old_flags['architecture']
            if architecture != flags.architecture:
                print(
                    'Warning: restored architecture does not agree with architecture specified in flags.')
            n_rec = old_flags['n_rec']
            if n_rec != flags.n_rec:
                print(
                    'Warning: restored number of recurrent units does not agree with number of recurrent units specified in flags.')

            model = MusicRNN(flags, architecture, n_rec)
            model.load_state_dict(torch.load(opj(flags.restore_from, 'results', 'model_checkpoint.pt')))

        else:
            model = MusicRNN(flags, flags.architecture, flags.n_rec)

        train_iter, valid_iter, test_iter = get_datasets(flags)

        train_loop(sm, flags, model, train_iter, valid_iter, test_iter)


if __name__ == '__main__':

    # system
    absl.flags.DEFINE_bool(
        'use_gpu', False, 'Whether or not to use the GPU. Fails if True and CUDA is not available.')
    absl.flags.DEFINE_string(
        'exp_name', '', 'If non-empty works as a special directory for the experiment')
    absl.flags.DEFINE_string('result_dir', 'trained_models',
                             'Name of the directory to save all results within.')
    absl.flags.DEFINE_integer(
        'random_seed', -1, 'If not negative 1, set the random seed to this value. Otherwise the random seed will be the current microsecond.')

    # training
    absl.flags.DEFINE_enum('dataset', 'JSB_Chorales', [
                           'JSB_Chorales', 'Nottingham', 'Piano_midi', 'MuseData'], 'Which dataset to train the model on.')
    absl.flags.DEFINE_integer('train_iters', 20000,
                              'How many training batches to show the network.')
    absl.flags.DEFINE_integer('batch_size', 100, 'Batch size.')
    absl.flags.DEFINE_float('lr', 0.001, 'Learning rate.')
    absl.flags.DEFINE_integer(
        'decay_every', 1000, 'Shrink the learning rate after this many training steps.')
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
    absl.flags.DEFINE_integer(
        'validate_every', 100, 'Validate the model at this many training steps.')
    absl.flags.DEFINE_integer(
        'save_every', 200, 'Save the model at this many training steps.')

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
