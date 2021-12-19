# CyberBach

Training machines to make music!

Complete with 5 pre-trained models and 20 whacky pre-generated tunes! Read about it in detail in [my blog post](https://ebanflo42.github.io/resources/docs/blog/cyberbach.html).

## Usage

Assuming you have cloned the repository and have a working installation of `conda` run

```
conda env create -f environment.yml
```

`conda` will most likely fail to install the IGI simulation manager, which can be installed from GitHub with `pip`:

```
pip install https://github.com/IGITUGraz/SimManager/archive/v0.8.3.zip
```

Create new directories for storing models and songs:

```
mkdir models songs
```

To create a new song from the pre-trained limit cycle GRU with input from the Nottingham dataset, run:

```
python make_music.py --model_path=cyberbach_models/gru_limitcycle --dataset=Nottingham
```

If you have [timidity](http://timidity.sourceforge.net/) installed, you can pass the `--use_timidity` flag to automatically convert the midi output to wav.

To train a GRU with a limit cycle initialization, run:

```
python train_model.py --architecture=GRU --initialization=limit_cycle
```

If you have a CUDA-capable GPU and CUDA-capable PyTorch installation, you can pass `--use_gpu` to greatly accelerate training. You can also pass the `--plot` flag to see some details of the models prediction and the dynamics of its hidden states.

All flags are documented at the top of each script.

## About

Recurrent neural networks can be trained to predict the next set of notes in piano sheet music. Here, neural networks are first trained to predict piano music and then used to construct a new tune by iteratively predicting the next notes of a certain input song.

There are four classic music datasets in `locuslab_data`, pre-processed from midi to piano roll by locuslab. `cyberbach_models` contains 5 models pre-trained for 10000 iterations on the `JSB_Chorales` (Bach's chorales) dataset; songs which these models synthesized is located in `cyber_bach` songs.

This framework supports 3 architectures: `TANH` (Vanilla), `LSTM`, and `GRU`. The three supported initialization strategies are `default` (Xavier), `orthogonal`, and `limit_cycle`. Limit cycle initialization is an idea for the `TANH` and `GRU` architectures which is introduced in my paper with CATNIP Lab:

P. A. Sokół, I. Jordan, E. Kadile and I. M. Park, "Adjoint Dynamics of Stable Limit Cycle Neural Networks," 2019 53rd Asilomar Conference on Signals, Systems, and Computers, 2019, pp. 884-887, doi: 10.1109/IEEECONF44664.2019.9049080.

In the paper we prove that a recurrent neural network initialized to a limit cycle will have stable gradients. Empirically, I found that initializing near the bifurcation point of limit cycle and attracting fixed point is beneficial for performance, and this is the special intiailization implemented in `utils.initialization.py`.