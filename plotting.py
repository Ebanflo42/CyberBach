import numpy as np
import numpy.linalg as la
import scipy.io as io
import torch
import json

import subprocess
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Custom colormap
cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.5, 0.0, 0.0),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
mymap = LinearSegmentedColormap('MyMap', cdict)
plt.register_cmap(cmap=mymap)

def plot_scalar(dir: str, name: str):
    """
    :param dir: directory of the file storage system whose results we are looking at
    :param name: name of the scalar metric we want to visualize
    """

    path = 'results/' + dir + '/'
    handle = open(path + 'metrics.json')
    content = handle.read()
    handle.close()

    json_dict = json.loads(content)
    values = json_dict[name]['values']

    plt.plot(values)
    plt.title(name + ' ' + dir)
    plt.show()


def plot_hidden_weights(dir: str, name: str, vmin: float, vmax: float):
    """
    :param dir: directory of the file storage system whose results we are looking at
    :param name: name of the .pt file whose .weight_hh_l0.weight we will visualize
    :param vmin: expected minimum weight
    :param vmax: expected maximum weight
    """

    path = 'results/' + dir + '/'

    sd = torch.load(path + name, map_location='cpu')
    hidden_weights = sd['rnn.weight_hh_l0.weight'].detach().numpy()

    #plt.title(name + ' weights ' + dir)
    fig, ax = plt.subplots()
    ax.pcolor(hidden_weights, vmin=vmin, vmax=vmax, cmap='MyMap')
    ax.set_aspect('equal')
    fig.show()
    plt.gca().invert_yaxis()


def plot_eigs(dir: str, name: str, lim: float):
    """
    :param dir: directory of the file storage system whose results we are looking at
    :param name: name of the .pt file whose .weight_hh_l0.weight eigenvalues we will visualize
    :param lim: how large is the square defining the plot
    """

    path = 'results/' + dir + '/'

    sd = torch.load(path + name, map_location='cpu')
    hidden_weights = sd['rnn.weight_hh_l0.weight'].detach().numpy()

    vals, vecs = la.eig(hidden_weights)

    fig, ax = plt.subplots()
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.scatter(np.real(vals), np.imag(vals))
    ax.set_aspect('equal')
    fig.show()


def get_metrics(dirs: str, metric: str):
    """
    :param dirs: list of directories for which we will look for the final metric
    :param metric: name of the metric we are going to plot
    :return: list of metrics after training
    """

    result = []

    for name in dirs:

        handle = open('results/' + name + '/metrics.json')
        my_dict = json.loads(handle.read())
        handle.close()

        result.append(my_dict[metric]['values'][-1])

    return result


def make_bar(labels, accuracy, loss):
    """
    :param labels: name of the models with number of parameters
    :param accuracy: accuracy achieved by each model at the end of training
    :param loss: loss achieved by each model at the end of training
    """

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, loss, width, label='Loss')

    ax.tick_params(axis='x', which='major', labelsize=6)
    ax.tick_params(axis='x', which='minor', labelsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(str(height)[0 : 5],
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(str(height)[0 : 5],
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()

    plt.show()