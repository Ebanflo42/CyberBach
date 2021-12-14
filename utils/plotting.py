import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from os.path import join as opj


def plot_note_comparison(sm, out_logits, targ, plot_i):

    # convert to numpy and get the first sample from the batch
    targ = targ.detach().cpu().numpy()[0]
    targ_bin = targ.astype(np.bool)

    out_logits = out_logits.detach().cpu().numpy()[0]
    out_bin = out_logits > 0
    out_int = out_bin.astype(np.int64)

    fig = plt.figure()
    ax = fig.add_subplot()

    T = 0
    for t in range(targ_bin.shape[0]):
        if np.any(targ_bin[t]):
            T = t

    low_key = 0
    for k in range(88):
        if np.sum(out_int[:, k] + targ[:, k]) == 0:
            low_key = k
        else:
            break
    high_key = 88
    for k in range(87, -1, -1):
        if np.sum(out_int[:, k] + targ[:, k]) == 0:
            high_key = k
        else:
            break

    ax.set_yticks([k for k in range(0, 88, 11)])
    ax.set_xticks([t for t in range(0, T, 10)])

    for t in range(T):
        for k in range(low_key, high_key):
            if targ_bin[t, k] and not out_bin[t, k]:
                ax.add_patch(Rectangle((t, k), 1, 1, facecolor='white', edgecolor='black'))
            elif targ_bin[t, k] and out_bin[t, k]:
                ax.add_patch(Rectangle((t, k), 1, 1, facecolor='green', edgecolor='black', fill=True))
            elif not targ_bin[t, k] and out_bin[t, k]:
                ax.add_patch(Rectangle((t, k), 1, 1, facecolor='red', fill=True))

    ax.set_xlabel('Time (beats)')
    ax.set_ylabel('Note')
    ax.set_title('Target notes compared to network prediction')

    plt.savefig(opj(sm.paths.results_path, f'note_comparison{plot_i}.png'))


def plot_phase_portrait(sm, model, plot_i):

    fig = plt.figure()
    ax = fig.add_subplot()

    X = np.linspace(-1, 1, 30)
    Y = np.linspace(-1, 1, 30)

    if model.architecture == 'TANH':

        w = model.rnn.weight_hh_l0.detach().cpu().numpy()[:2, :2]
        b = model.rnn.bias_hh_l0.detach().cpu().numpy()[:2]

        u, v = np.zeros((30, 30)), np.zeros((30, 30))

        for i in range(30):
            for j in range(30):
                xy = np.stack([X[i], Y[j]])
                uv = np.tanh(w@xy + b) - xy
                u[i, j] = uv[0]
                v[i, j] = uv[1]

        ax.streamplot(X, Y, v, u, linewidth=0.5)

    elif model.architecture == 'GRU':

        def sigmoid(x):
            return 1/(1 + np.exp(-x))

        whh = model.rnn.weight_hh_l0.detach().cpu().numpy()
        bhh = model.rnn.bias_hh_l0.detach().cpu().numpy()

        whr = whh[:2, :2]
        whz = whh[model.n_rec : model.n_rec + 2, :2]
        whn = whh[2*model.n_rec : 2*model.n_rec + 2, :2]

        bhr = bhh[:2]
        bhz = bhh[model.n_rec : model.n_rec + 2]
        bhn = bhh[2*model.n_rec : 2*model.n_rec + 2]

        u, v = np.zeros((30, 30)), np.zeros((30, 30))

        for i in range(30):
            for j in range(30):
                xy = np.stack([X[i], Y[j]])
                r = sigmoid(whr@xy + bhr)
                z = sigmoid(whz@xy + bhz)
                n = np.tanh(r*(whn@xy + bhn))
                uv = (1 - z)*n + z*xy - xy
                u[i, j] = uv[0]
                v[i, j] = uv[1]

        ax.streamplot(X, Y, v, u, linewidth=0.5)

    else:
        raise ValueError(f'Phase portrait plotting not implemented for architecture {model.architecture}.')

    ax.set_title('Autonomous phase portrait of first two hidden states')

    plt.savefig(opj(sm.paths.results_path, f'phase_portrait{plot_i}.png'))