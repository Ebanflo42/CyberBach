"""
These functions are used for translating between midi and the binary piano roll format,
as well as synthesizing new music.
"""

import numpy as np
import torch
import mido
import os

from tqdm import tqdm
from os.path import join as opj


def get_min_max_note(directory: str):
    """
    :param directory: name of directory containing a bunch of midi files
    :return: minimum and maximum note found in all files
    """

    min_note = 255
    max_note = 0

    for filename in os.listdir(directory):
        if filename.endswith('.mid'):
            midfile = mido.MidiFile(opj(directory,  filename))
            for msg in midfile:
                if msg.type == 'note_on':
                    note = msg.note
                    if note > max_note:
                        max_note = note
                    if note < min_note:
                        min_note = note

    return min_note, max_note


def get_msg_list(midi):
    """
    :param midi: a parsed midi track
    :return: list of all on/off signals (filters meta signals, etc)
    """
    result = []
    for msg in midi:
        if msg.type == 'note_on' or msg.type == 'note_off':
            result.append(msg)
    return result


def get_increment(msg_list):
    """
    :param msg_list: list of on/off midi signals
    :return: minimum time of all signals
    """
    result = float('inf')
    for msg in msg_list:
        if msg.time != 0.0 and msg.time < result:
            result = msg.time
    return result


def to_piano_roll(in_filename: str, min_note: int, max_note: int):
    """
    :param in_filename: name of midi file to be converted to piano roll
    :param min_note: minimum note that might be contained in the file
    :param max_note: maximum note that might be contained in the file
    :return: binary numpy array whose first index is time and whose second index is note id - element will be 1 if the note is on at that time and 0 otherwise.
    """

    midi = mido.MidiFile(in_filename)
    msg_list = get_msg_list(midi)
    increment = get_increment(msg_list)

    array = np.zeros((int(midi.length/increment) + 1, max_note - min_note + 1), dtype='uint8')

    index = 0
    on_notes = []

    for msg in msg_list:

        time = int(msg.time/increment)

        if msg.type == 'note_on':
            on_notes.append((msg.note - min_note, msg.velocity))
        elif msg.type == 'note_off':
            on_notes = [(n, v) for (n, v) in on_notes if n != msg.note - min_note]

        for t in range(index, index + time + 1):
            for note, velocity in on_notes:
                array[t, note] = 1

        index += time

    return array


def to_midi(min_note, piano_roll_song, filename):

    # create a new midi file with a single track
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    #track.append(mido.Message('program_change', program=12, time=0))

    track.append(mido.MetaMessage('key_signature', key='C', time=0))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    #track.append(mido.MetaMessage('set_tempo', tempo=705882))

    # keep track of which notes are being played at the given time
    on_notes = []

    # keep track of how many notes have passed since a change was made
    since_change = 1

    # record all messages since the last change
    # when a change occurs, append them with the last one indicating the amount of time passed
    old_messages = []
    new_messages = []

    # loop through time
    for i in range(piano_roll_song.shape[0]):

        # did anything change at this time step
        nothing_happened = True

        # if this is the last time step, that counts as something happening
        if i == piano_roll_song.shape[0] - 1:
            nothing_happened = False

        # loop through notes
        for j in range(piano_roll_song.shape[1]):

            # if this note is being played at the current time
            if piano_roll_song[i, j] == 1:

                # if it was not being played before
                if not j in on_notes:

                    # this new message indicates the note was turned on
                    new_msg = mido.Message('note_on', note=min_note+j, velocity=90, time=0)

                    # it is a new message so we append it to the appropriate list
                    new_messages.append(new_msg)

                    # something happened, this note was turned on
                    nothing_happened = False

                    # record that this note is being played
                    on_notes.append(j)

                # otherwise do nothing
                else:
                    continue

            # if this note is not being played at the current time
            else:

                # if it was being played
                if j in on_notes:

                    # this new message indicates the note was turned off
                    new_msg = mido.Message('note_off', note=min_note+j, velocity=0, time=0)

                    # it is a new message so we append it to the appropriate list
                    new_messages.append(new_msg)

                    # something happened, this note was turned off
                    nothing_happened = False

                    # remove the note from on_notes
                    on_notes = [note for note in on_notes if not note == j]

                # otherwise do nothing
                else:
                    continue

        # if nothing happened simply increment the time since something happened
        if nothing_happened:
            since_change += 1

        # if something happened, append all the messages that have been accumulating
        # and replace the old messages with the new ones
        else:

            # if this is the first time step there will be no old messages
            if len(old_messages) > 0:

                # make the last old message lag the appropriate time
                old_messages[-1].time = 120*since_change

                # append the old messages
                for msg in old_messages:
                    track.append(msg)

                # erase old messages and replace them with new, reset counter
                old_messages = new_messages
                new_messages = []
                since_change = 1

            else:

                # we need to wait to append them until we know how much time has passed
                old_messages = new_messages
                new_messages = []

    # save the file
    mid.save(filename)


def write_song(model, piano_roll, FLAGS):
    """
    :param model: pytorch model which will be used to synthesize new music
    :param piano roll: 2D binary array indicating which notes are played at a given time
    :param true_steps: how many frames of the new song are to be copied directly from the original
    :param input_steps: how many frames of the new song are to be the output of the model being fed piano_roll
    :param free_steps: how many frames of the new song are to be the model predicting the next frame based off of the song constructed so far
    :param history: how many time steps the model will look back in the past while constructing the new song
    :param variance: variance of the noise to be added to the hidden state (in order to avoid stable periodic orbits)
    """

    # first few steps of the song will be the original music
    T1 = len(piano_roll)
    T = T1 + FLAGS.free_steps
    song = np.zeros((T, 88), dtype='uint8')

    # format the input to the model
    input_tensor = torch.tensor(piano_roll, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)

    # format the output of the model
    output_tensor, hiddens = model(input_tensor)
    np_output = output_tensor.detach().numpy()
    binary = (np_output > 0).astype(np.uint8)
    reformatted = binary.reshape(T1, 88)

    # check for too many or too few notes being played
    # adjust the threshold for the logits if the number of notes falls outside the desired range
    for t in range(T1):
        if np.sum(reformatted[t]) > FLAGS.max_on_notes:
            threshold = 0
            while np.sum((np_output[t] > threshold).astype(np.uint8)) > FLAGS.max_on_notes:
                threshold += 1
            reformatted[t] = (np_output[t] > threshold).astype(np.uint8)
        if np.sum(reformatted[t]) < FLAGS.min_on_notes:
            threshold = 0
            while np.sum((np_output[t] > threshold).astype(np.uint8)) < FLAGS.min_on_notes:
                threshold -= 1
            reformatted[t] = (np_output[t] > threshold).astype(np.uint8)

    song[:T1] = reformatted

    # the last steps of the model will be the model making predictions off of its own output
    for t in tqdm(range(T1, T)):

        # get the last frame of the new song, double the intensity since it is both input and last step
        last_output = torch.tensor(song[t - 1], dtype=torch.float32).unsqueeze(0)
        last_output += FLAGS.noise_variance*torch.randn((1, 1, 88))

        # use the model to predict the next frame
        new_output, hiddens = model(last_output)
        np_output = new_output.detach().numpy().reshape(88)
        binary = (new_output > 0).astype(np.uint8)

        # again adjust the threshold if there are too many notes
        if np.sum(binary[t]) > FLAGS.max_on_notes:
            threshold = 0
            while np.sum((np_output[t] > threshold).astype(np.uint8)) > FLAGS.max_on_notes:
                threshold += 1
            binary[t] = (np_output[t] > threshold).astype(np.uint8)
        if np.sum(binary[t]) < FLAGS.min_on_notes:
            threshold = 0
            while np.sum((np_output[t] > threshold).astype(np.uint8)) < FLAGS.min_on_notes:
                threshold -= 1
            binary[t] = (np_output[t] > threshold).astype(np.uint8)

        song[t] = binary

    return song