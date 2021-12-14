import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from os.path import join as opj


def plot_note_comparison(sm, out_logits, targ, plot_i):

    # convert to numpy and get the first sample from the batch
    targ_bin = targ.detach().cpu().numpy()[0].astype(np.bool)
    out_bin = (out_logits.detach().cpu().numpy()[0] > 0)
    out_int = out_bin.astype(np.int64)

    fig = plt.figure()
    ax = fig.add_subplot()

    T = targ_bin.shape[0]

    low_key = 0
    for k in range(88):
        if np.sum(out_int[:, k] + targ[:, k]) == 0:
            low_key = k
        else:
            break
    high_key = 88
    for k in range(88, 0, -1):
        if np.sum(out_int[:, k] + targ[:, k]) == 0:
            high_key = k
        else:
            break

    for t in range(T):
        for k in range(low_key, high_key):
            if targ_bin[t, k] and not out_bin[t, k]:
                ax.add_patch(Rectangle((t, k), 1, 1), facecolor='white', edgecolor='black')
            elif targ_bin[t, k] and out_bin[t, k]:
                ax.add_patch(Rectangle((t, k), 1, 1), facecolor='green', edgecolor='black', fill=True)
            elif not targ_bin[t, k] and out_bin[t, k]:
                ax.add_patch(Rectangle((t, k), 1, 1), facecolor='red', fill=True)

    ax.set_xlabel('Time (beats)')
    ax.set_ylabel('Note')
    ax.set_title('Target notes compared to network prediction')

    plt.savefig(opj(sm.paths.results_path, f'note_comparison{plot_i}.png'))