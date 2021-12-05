"""
This script is for generating new music based on the LocusLab datasets and the models trained on them.
"""

import json
import torch
import simmanager
import subprocess

from absl import app, flags
from scipy.io import loadmat
from os.path import join as opj
from random_word import RandomWords

from utils.models import MusicRNN
from utils.midi import to_midi, write_song


FLAGS = flags.FLAGS

# file system
flags.DEFINE_string('results_path', 'songs', 'Where to store new songs.')
flags.DEFINE_string('exp_name', '', 'If non-empty acts as a special subdirectory for a set of songs.')
flags.DEFINE_string('song_name', '', 'Optional song name. If empty, random song name will be generated.')
flags.DEFINE_boolean('use_timidity', False, 'Use timidity to convert the midi file to wav. Fails if timidity is not installed.')

# where to draw input
flags.DEFINE_enum('dataset', 'JSB_Chorales', ['JSB_Chorales', 'Nottingham', 'Piano_midi', 'MuseData'], 'Which dataset to base a song off of.')
flags.DEFINE_enum('subset', 'train', ['train', 'valid', 'test'], 'Which subset to grab a song to synthesize from')
flags.DEFINE_integer('index', 0, 'Index of the input song in the dataset.')
flags.DEFINE_integer('beat', 0, 'Which beat to start the model on.')

# song synthesis
flags.DEFINE_string('model_path', '', 'Which model to restore to use to synthesize song. If empty, output will be the original song')
flags.DEFINE_integer('free_steps', 100, 'How many beats we should continue after the network has been fed the entire song.')
flags.DEFINE_integer('max_on_notes', 10, 'Maximum number of notes to be played during a beat.')
flags.DEFINE_integer('min_on_notes', 0, 'Minimum number of notes to be played during a beat.')
flags.DEFINE_float('noise_variance', 0, 'Gaussian noise may be added to the model input to knock it out of periodic behavior.')


def main(_argv):

    # construct simulation manager
    base_path = opj(FLAGS.results_path, FLAGS.exp_name)
    if FLAGS.song_name != '':
        song_name = FLAGS.song_name
    else:
        r = RandomWords()
        rws = r.get_random_words()
        song_name = rws[0] + '_' + rws[1]
    print(f'Beginning new song {song_name}.')
    sm = simmanager.SimManager(
        song_name, base_path, write_protect_dirs=False, tee_stdx_to='output.log')

    with sm:

        # check for timidity
        if FLAGS.use_timidity and not subprocess.call('timidity') == 0:
            raise OSError(
                '`timidity` is not installed?')

        # dump FLAGS for this experiment
        with open(opj(sm.paths.data_path, 'FLAGS.json'), 'w') as f:
            flag_dict = {}
            for k in FLAGS._flags().keys():
                if k not in FLAGS.__dict__['__hiddenflags']:
                    flag_dict[k] = FLAGS.__getattr__(k)
            json.dump(flag_dict, f)

        # check for old model to restore from
        model = None
        if FLAGS.model_path != '':

            with open(opj(FLAGS.model_path, 'data', 'FLAGS.json'), 'r') as f:
                old_FLAGS = json.load(f)

            architecture = old_FLAGS['architecture']
            n_rec = old_FLAGS['n_rec']

            model = MusicRNN(architecture, n_rec)
            model.load_state_dict(torch.load(
                opj(FLAGS.model_path, 'results', 'model_checkpoint.pt')))

        # load music sample
        matdata = loadmat(f'locuslab_data/{FLAGS.dataset}.mat')
        piano_roll = matdata[f'{FLAGS.subset}data'][0][FLAGS.index]

        # use model to synthesize new music
        if model is not None:
            new_piano_roll = write_song(model, piano_roll, FLAGS)
            to_midi(0, new_piano_roll, opj(sm.paths.results_path, song_name + '.mid'))
        else:
            to_midi(0, piano_roll, opj(sm.paths.results_path, song_name + '.mid'))

        if FLAGS.use_timidity:
            subprocess.call(f'timidity -Ow {opj(sm.paths.results_path, song_name)}.mid', shell=True)


if __name__ == '__main__':

    names = ['JSB_Chorales', 'Nottingham', 'Piano_midi', 'MuseData']
    for name in names:
        avglen = 0
        matdata = loadmat(f'locuslab_data/{name}.mat')
        for train_sample in matdata['traindata'][0]:
            avglen += len(train_sample)
        print(name, avglen/len(matdata['traindata'][0]))

    app.run(main)