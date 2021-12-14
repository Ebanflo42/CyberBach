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
flags.DEFINE_string('song_name', '', 'Optional song name. If empty, random song name will be generated.')
flags.DEFINE_boolean('use_timidity', False, 'Use timidity to convert the midi file to wav. Fails if timidity is not installed.')

# where to draw input
flags.DEFINE_enum('dataset', 'JSB_Chorales', ['JSB_Chorales', 'Nottingham', 'Piano_midi', 'MuseData'], 'Which dataset to base a song off of.')
flags.DEFINE_enum('subset', 'train', ['train', 'valid', 'test'], 'Which subset to grab a song to synthesize from')
flags.DEFINE_integer('index', 0, 'Index of the input song in the dataset.')

# song synthesis
flags.DEFINE_string('model_path', '', 'Which model to restore to use to synthesize song. If empty, output will be the original song')
flags.DEFINE_integer('length', 200, 'How many beats the song should last.')
flags.DEFINE_integer('max_on_notes', 10, 'Maximum number of notes to be played during a beat.')
flags.DEFINE_integer('min_on_notes', 0, 'Minimum number of notes to be played during a beat.')
flags.DEFINE_float('noise_variance', 0.05, 'Gaussian noise may be added to the model input to knock it out of periodic behavior.')


def make_song_name():
    try:
        rw = RandomWords()
        rws = rw.get_random_words()
        good_rws = [w for w in rws if '\'' not in w and ' ' not in w]
        return good_rws[0] + '_' + good_rws[1]
    except TypeError:
        print('Warning: `make_song_name` failed. Trying again.')
        return make_song_name()


def main(_argv):

    # construct simulation manager
    if FLAGS.song_name != '':
        song_name = FLAGS.song_name
    else:
        song_name = make_song_name()
    print(f'Beginning new song {song_name}.')
    sm = simmanager.SimManager(
        song_name, FLAGS.results_path, write_protect_dirs=False, tee_stdx_to='output.log')

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

    app.run(main)