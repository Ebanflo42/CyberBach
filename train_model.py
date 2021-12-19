import json
import string
import random
import pickle as pkl

import simmanager
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.models import MusicRNN
from utils.metrics import FrameAccuracy, MaskedBCE
from utils.data_loader import get_datasets
from utils.initialization import initialize
from utils.plotting import plot_note_comparison, plot_phase_portrait

from absl import flags, app
from datetime import datetime
from os.path import join as opj


FLAGS = flags.FLAGS

# system
flags.DEFINE_bool(
    'use_gpu', False, 'Whether or not to use the GPU. Fails if True and CUDA is not available.')
flags.DEFINE_string(
    'model_name', '', 'If non-empty works as a special name for this model.')
flags.DEFINE_string('results_path', 'models',
                    'Name of the directory to save all results within.')
flags.DEFINE_integer(
    'random_seed', -1, 'If not -1, set the random seed to this value. Otherwise the random seed will be the current microsecond.')

# training
flags.DEFINE_enum('dataset', 'JSB_Chorales', [
    'JSB_Chorales', 'Nottingham', 'Piano_midi', 'MuseData'], 'Which dataset to train the model on.')
flags.DEFINE_integer('n_steps', 10000,
                     'How many training batches to show the network.')
flags.DEFINE_integer('batch_size', 50, 'Batch size.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_integer(
    'decay_every', 1000, 'Shrink the learning rate after this many training steps.')
flags.DEFINE_float(
    'lr_decay', 0.95, 'Shrink the learning rate by this factor.')
flags.DEFINE_enum('optimizer', 'RMSprop', [
    'Adam', 'SGD', 'Adagrad', 'RMSprop'], 'Which optimizer to use.')
flags.DEFINE_float('reg_coeff', 0.0001,
                   'Coefficient for L2 regularization of weights.')
flags.DEFINE_bool(
    'use_grad_clip', False, 'Whether or not to clip the backward gradients by their magnitude.')
flags.DEFINE_float(
    'grad_clip', 1, 'Maximum magnitude of gradients if gradient clipping is used.')
flags.DEFINE_integer(
    'validate_every', 500, 'Validate the model at this many training steps.')
flags.DEFINE_integer(
    'save_every', 1000, 'Save the model at this many training steps.')
flags.DEFINE_boolean('plot', False, 'Plot note comparison and phase portrait every validation step.')

# model
flags.DEFINE_enum('architecture', 'TANH', [
    'TANH', 'LSTM', 'GRU'], 'Which recurrent architecture to use.')
flags.DEFINE_integer(
    'n_rec', 400, 'How many recurrent neurons to use.')
flags.DEFINE_enum('initialization', 'default', ['default', 'orthogonal', 'limit_cycle'],
                  'Which initialization to use for the recurrent weight matrices. Default is uniform Xavier. Limit cycles only apply to TANH and GRU.')
flags.DEFINE_string(
    'restore_from', '', 'If non-empty, restore the previous model from this directory and train it using the new flags.')


def train_loop(sm, FLAGS, model, train_iter, valid_iter, test_iter):

    # if we are on cuda we construct the device and run everything on it
    device = torch.device('cpu')
    if FLAGS.use_gpu:
        dev_name = 'cuda:0'
        device = torch.device(dev_name)
        model = model.to(device)

    train_loss = []
    train_reg = []
    train_acc = []

    valid_loss = []
    valid_acc = []

    # construct the optimizer
    if FLAGS.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr)
    elif FLAGS.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    elif FLAGS.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=FLAGS.lr)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=FLAGS.lr)
    else:
        raise ValueError(f'Optimizer {FLAGS.optimizier} not recognized.')

    # learning rate decay
    scheduler = None
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: FLAGS.lr_decay**(epoch//FLAGS.decay_every))

    acc_fcn = FrameAccuracy()
    loss_fcn = MaskedBCE()

    # begin training loop
    for i in range(FLAGS.n_steps):

        # get next training sample
        x, y, mask = next(train_iter)
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        # forward pass
        output, hidden = model(x)

        # binary cross entropy
        bce_loss = loss_fcn(output, y, mask)

        # weight regularization
        l2_reg = torch.tensor(0, dtype=torch.float32, device=device)
        for param in model.parameters():
            l2_reg += FLAGS.reg_coeff*torch.norm(param)

        # backward pass and optimization step
        loss = bce_loss + l2_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # learning rate decay
        scheduler.step()

        # compute accuracy
        acc = acc_fcn(output, y, mask)

        # append metrics
        train_loss.append(bce_loss.cpu().item())
        train_acc.append(acc.cpu().item())
        train_reg.append(l2_reg.cpu().item())

        if i > 0 and i % FLAGS.validate_every == 0:

            timestamp = datetime.now().strftime('%H:%M:%S')
            print(
                f'{timestamp} Validating {sm.sim_name} at iteration {i}.\n  Training loss: {train_loss[-1]:.3}\n  Training accuracy: {100*train_acc[-1]:.3}%\n  L2 regularization: {train_reg[-1]:.3}')

            # get next validation sample
            x, y, mask = next(valid_iter)
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            # forward pass
            output, hidden = model(x)

            # binary cross entropy
            bce_loss = loss_fcn(output, y, mask)

            # compute accuracy
            acc = acc_fcn(output, y, mask)

            # append metrics
            valid_loss.append(bce_loss.cpu().item())
            valid_acc.append(acc.cpu().item())

            print(
                f'  Validation loss: {valid_loss[-1]:.3}\n  Validation accuracy: {100*valid_acc[-1]:.3}%\n')

            if FLAGS.plot:
                plot_note_comparison(sm, output, y, i)
                if model.architecture in ['TANH', 'GRU']:
                    plot_phase_portrait(sm, model, i)

        if i > 0 and i % FLAGS.save_every == 0:

            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f'{timestamp} Saving {sm.sim_name} at iteration {i}.\n')

            np.save(opj(sm.paths.results_path, 'training_loss'), train_loss)
            np.save(opj(sm.paths.results_path,
                    'training_accuracy'), train_acc)
            np.save(opj(sm.paths.results_path,
                    'training_regularization'), train_reg)

            np.save(opj(sm.paths.results_path, 'validation_loss'), valid_loss)
            np.save(opj(sm.paths.results_path,
                    'validation_accuracy'), valid_acc)

            torch.save(model.state_dict(), opj(
                sm.paths.results_path, 'model_checkpoint.pt'))

    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'{timestamp} Finished training. Entering testing phase.')

    test_loss = []
    test_acc = []
    tot_test_samples = 0

    # loop through entire testing set
    for x, y, mask in test_iter:

        x, y, mask = x.to(device), y.to(device), mask.to(device)

        output, hidden = model(x)

        bce_loss = loss_fcn(output, y, mask).cpu().item()
        acc = acc_fcn(output, y, mask).cpu().item()

        batch_size = x.shape[0]
        tot_test_samples += batch_size
        test_loss.append(batch_size*bce_loss)
        test_acc.append(batch_size*acc)

    final_test_loss = np.sum(test_loss)/tot_test_samples
    final_test_acc = np.sum(test_acc)/tot_test_samples
    print(
        f'  Testing loss: {final_test_loss:.3}\n  Testing accuracy: {100*final_test_acc:.3}%')

    print(f'Final save of.')
    np.save(opj(sm.paths.results_path, 'testing_loss'), final_test_loss)
    np.save(opj(sm.paths.results_path, 'testing_accuracy'), final_test_acc)
    torch.save(model.state_dict(), opj(
        sm.paths.results_path, 'model_checkpoint.pt'))
    date = datetime.now().strftime('%d-%m-%Y')
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'Training {sm.sim_name} ended on {date} at {timestamp}.')


def main(_argv):

    # construct simulation manager
    if FLAGS.model_name == '':
        identifier = ''.join(random.choice(
            string.ascii_lowercase + string.digits) for _ in range(4))
        sim_name = 'model_{}'.format(identifier)
    else:
        sim_name = FLAGS.model_name
    date = datetime.now().strftime('%d-%m-%Y')
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f'\nBeginning to train {sim_name} on {date} at {timestamp}.\n')
    sm = simmanager.SimManager(
        sim_name, FLAGS.results_path, write_protect_dirs=False, tee_stdx_to='output.log')

    with sm:

        # check for cuda
        if FLAGS.use_gpu and not torch.cuda.is_available():
            raise OSError(
                'CUDA is not available. Check your installation or set `use_gpu` to False.')

        # dump FLAGS for this experiment
        with open(opj(sm.paths.data_path, 'FLAGS.json'), 'w') as f:
            flag_dict = {}
            for k in FLAGS._flags().keys():
                if k not in FLAGS.__dict__['__hiddenflags']:
                    flag_dict[k] = FLAGS.__getattr__(k)
            json.dump(flag_dict, f)

        # generate, save, and set random seed
        if FLAGS.random_seed != -1:
            random_seed = FLAGS.random_seed
        else:
            random_seed = datetime.now().microsecond
        np.save(opj(sm.paths.data_path, 'random_seed'), random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # check for old model to restore from
        if FLAGS.restore_from != '':

            with open(opj(FLAGS.restore_from, 'data', 'FLAGS.json'), 'r') as f:
                old_FLAGS = json.load(f)

            architecture = old_FLAGS['architecture']
            if architecture != FLAGS.architecture:
                print(
                    'Warning: restored architecture does not agree with architecture specified in FLAGS.')
            n_rec = old_FLAGS['n_rec']
            if n_rec != FLAGS.n_rec:
                print(
                    'Warning: restored number of recurrent units does not agree with number of recurrent units specified in FLAGS.')

            model = MusicRNN(
                architecture, n_rec, use_grad_clip=FLAGS.use_grad_clip, grad_clip=FLAGS.grad_clip)
            model.load_state_dict(torch.load(
                opj(FLAGS.restore_from, 'results', 'model_checkpoint.pt')))

        else:
            model = MusicRNN(FLAGS.architecture, FLAGS.n_rec,
                             use_grad_clip=FLAGS.use_grad_clip, grad_clip=FLAGS.grad_clip)
            initialize(model, FLAGS)

        train_iter, valid_iter, test_iter = get_datasets(FLAGS)

        train_loop(sm, FLAGS, model, train_iter, valid_iter, test_iter)


if __name__ == '__main__':

    app.run(main)
