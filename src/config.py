import argparse

import json

from .models import MODELS, OPTS, LOSSES
from .data import DATASETS


WANDB_EXCLUDE_KEYS = ["name", "group", "notes", "project", "wandb_off"]


def load_defaults(parser, config_path):
    """
    Add defaults set in config file to parser. Useful for defining a second set
    of defaults more specific than the global defaults defined in this file.

    Ex: Putting together a audio_processing.json config for audio specific
    defaults like --dataset throat_mic_classif

    Json keys should be arguments (without '--') and values should be strings as
    they would be if input as command line args
    """
    if config_path is None:
        return
    with open(config_path, "r") as f:
        loaded_defaults = json.load(f)
    parser.set_defaults(**loaded_defaults)


# PARSING TYPES

def split_params(s):
    try:
        return [int(split) / 100 for split in s.split(',')]
    except:
        raise argparse.ArgumentError(
            "Argument must be of form train%,dev%,test%"
        )

def conv_params(s):
    try:
        layers = [l.split(",") for l in s.split()]
        return [(int(l),int(s),int(o)) for l,s,o in layers]
    except:
        raise argparse.ArgumentError(
            "Argument must be of form kernel_len,kernel_stride,out_size"
        )


def pool_params(s):
    try:
        pool_len, pool_stride = map(int, s.split(","))
        return pool_len, pool_stride
    except:
        raise argparse.ArgumentError(
            "Argument must be of form pool_len,pool_stride"
        )


# VALIDATION FUNCS

def validate_convs_pools(config):
    if config.pools is not None and len(config.convs) != len(config.pools):
        raise ValueError(
            "Equal number of conv layers and pooling layers must be defined. "
            f"{len(config.convs)} conv layers and"
            f"{len(config.pools)} pooling layers passed."
        )


def validate_aug_cache(config):
    if config.aug and config.cache_processed:
        raise ValueError(
            "Augmentations do not work with cache_processed! Use cache_raw or "
            "don't use augmentations"
        )


def validate_config(config):
    validate_convs_pools(config)
    validate_aug_cache(config)


def build_parsers():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=str,
        default=None,
        help=(
            "Path to config to load. Overwrites global defaults, but is"
            " overwritten by any command line arguments."
        )
    )

    parser = argparse.ArgumentParser(
        description="Train Signal Processing Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[config_parser]
    )

    parser.add_argument(
        "--sweep_support", type=int,
        default=0,
        help=(
            "Unused argument, meant for running multiple runs in a WandB grid "
            "search with the same arguments"
        )
    )

    ##########################################################################
    # WandB
    ##########################################################################
    parser.add_argument(
        "--entity", type=str,
        default="step-emg",
        help="WandB entity name"
    )
    parser.add_argument(
        "--project", type=str,
        default="Audio Signal Processing",
        help="WandB project name"
    )
    parser.add_argument(
        "--name", type=str,
        default=None,
        help="WandB run name"
    )
    parser.add_argument(
        "--group", type=str,
        default=None,
        help="Wandb run group"
    )
    parser.add_argument(
        "--notes", type=str,
        default=None,
        help="WandB run notes"
    )
    parser.add_argument(
        "--wandb_off", action='store_true',
        default=False,
        help="Turn WandB logging off"
    )

    ##########################################################################
    # Filepaths
    ##########################################################################
    parser.add_argument(
        "--data", type=str,
        default="data/processed_data/quiet_front-ear",
        help="Dataset to load. Should contain session dirs within"
    )
    parser.add_argument(
        "--save_dir", type=str,
        default="data/models/tmp/",
        help="Directory to save model to"
    )
    parser.add_argument(
        "--load_dir", type=str,
        default=None,
        help="Directory to load model and config from"
    )

    ##########################################################################
    # Dataset
    ##########################################################################
    parser.add_argument(
        "--dataset", type=str,
        default="throat_mic_classif",
        choices=sorted(set(DATASETS.keys())),
        help="Type of dataset to use."
    )
    parser.add_argument(
        "--cache_raw", type=int,
        default=1,
        help="Whether to load dataset into memory or not. 1 for True, 0 for False"
    )
    parser.add_argument(
        "--cache_processed", type=int,
        default=0,
        help=(
            "Whether to load processed dataset into memory or not. 1 for True, "
            "0 for False. WARNING: WILL BREAK AUGMENTATIONS"
        )
    )
    parser.add_argument(
        "--norm", type=int,
        default=0,
        help="Whether to normalize inputs or not. 1 for True 0 for False."
    )

    ##### Splits #####
    parser.add_argument(
        "--seed", type=int,
        default=42,
        help="Random seed to use for splits"
    )
    parser.add_argument(
        "--splits", type=split_params,
        default=(0.7, 0.15, 0.15),
        help=(
            "What splits to use for the train / dev / test sets. Should be"
            "specified with each percentage expressed as an integer, seperated"
            " by commas. Ex: To recreate the default splits, "
            "'--splits 70,15,15'"
        )
    )
    parser.add_argument(
        "--split_sessions", type=int,
        default=0,
        help=(
            "Whether to split the data on sessions or not. When enabled,"
            "holdout data sessions will never be trained on. "
            "1 for True, 0 for False"
        )
    )
    parser.add_argument(
        "--split_subjects", type=int,
        default=0,
        help=(
            "Whether to split the data on subjects or not. When enabled,"
            "holdout data subjects will never be trained on. "
            "1 for True, 0 for False"
        )
    )
    parser.add_argument(
        "--stratify", type=int,
        default=1,
        help=(
            "Whether to use stratified sampling for train / dev / test split."
            "Guarentees even equal distribution of labels between splits, but"
            " breaks for small datasets. 1 for True, 0 for False"
        )
    )

    ##### Augmentation #####
    parser.add_argument(
        "--aug", type=int,
        default=0,
        help="Whether to use augmentations on train set. 1 for True, 0 for False"
    )
    parser.add_argument(
        "--aug_pad", type=int,
        default=0,
        help="Whether to use random padding. 1 for True, 0 for False"
    )
    parser.add_argument(
        "--aug_shift", type=int,
        default=0,
        help="Whether to use randomly shift data. 1 for True, 0 for False"
    )
    parser.add_argument(
        "--aug_volume", type=int,
        default=0,
        help="Max amount to vary normalized volume by. Disabled if 0"
    )
    parser.add_argument(
        "--aug_spec", type=int,
        default=0,
        help="Whether to use randomly mask spectogram. 1 for True, 0 for False"
    )

    ##### Length #####
    parser.add_argument(
        "--max_len", type=int,
        default=1024,
        help="Maximium length of sequence. Used in PaddedInputEncoder"
    )
    parser.add_argument(
        "--max_ms", type=int,
        default=3000,
        help="Maximium length in ms. Used in AudioInputEncoder"
    )

    ##### Audio Specific #####
    parser.add_argument(
        "--channels", type=int,
        default=1,
        help="Number of channels to use. Used in AudioLoader"
    )
    parser.add_argument(
        "--samplerate", type=int,
        default=48000,
        help="Sample rate to resample to. Used in AudioInputEncoder"
    )
    parser.add_argument(
        "--loudness", type=int,
        default=40,
        help="Loudness to normalize to. Disabled if 0. Used in AudioInputEncoder"
    )
    parser.add_argument(
        "--n_fft", type=int,
        default=1024,
        help="Size of FFT for mel spectogram. Used in AudioInputEncoder"
    )
    parser.add_argument(
        "--n_mels", type=int,
        default=64,
        help=(
            "Number of mel filterbanks for mel spectogram. Used in"
            " AudioInputEncoder"
        )
    )
    parser.add_argument(
        "--hop_len", type=int,
        default=None,
        help=(
            "Length of hop between windows in mel spectogram. Used in"
            " AudioInputEncoder"
        )
    )
    parser.add_argument(
        "--aug_spec_pct", type=float,
        default=0.10,
        help=(
            "Percentage of spectogram to mask when augmenting. "
            "Used in AudioInputEncoder"
        )
    )
    parser.add_argument(
        "--aug_spec_time", type=int,
        default=1,
        help=(
            "Number of time dimension masks to produce when augmenting"
            " spectograms. Used in AudioInputEncoder"
        )
    )
    parser.add_argument(
        "--aug_spec_freq", type=int,
        default=1,
        help=(
            "Number of freq dimension masks to produce when augmenting"
            " spectograms. Used in AudioInputEncoder"
        )
    )

    ##########################################################################
    # Model
    ##########################################################################
    parser.add_argument(
        "--model", type=str,
        default="2d_cnn",
        choices=sorted(set(MODELS.keys())),
        help="Type of model to train."
    )

    ##### Layers #####
    parser.add_argument(
        "--fcs", type=int, nargs='+',
        default=[128],
        help=(
            "Fully connected layers. Pass neurons per layer separated by "
            "spaces. Ex: For a model with 128 then 64 neurons, pass "
            "'--fcs 128 64'"
        )
    )
    parser.add_argument(
        "--convs", type=conv_params,
        default=[(5, 2, 8), (3, 2, 16), (3, 2, 32), (3, 2, 64)],
        help=(
            "CNN layers. Pass kernel size, stride and output channels seperated"
            " by commas, with layers separated by spaces. Ex: For a model with "
            "two layers of kernel size 3, stride 1, out channel 8, pass "
            '--convs "3,1,8 3,1,8"'
        )
    )
    parser.add_argument(
        "--pools", type=pool_params, nargs='+',
        default=None,
        help=(
            "Pooling layers. Pass pool size and stride seperated by commas, "
            "with layers separated by spaces. Ex: For a model with "
            "two layers of pooling of size 3, stride 1 pass "
            "'--pools 3,1 3,1'"
        )
    )
    parser.add_argument(
        "--adaptive_pool", type=int,
        default=0,
        help=(
            "Whether to pool final layers down to single values."
            "1 for True, 0 for False"
        )
    )

    ##### Regularization #####
    parser.add_argument(
        "--drop_prob", type=float,
        default=0.25,
        help="Dropout probability."
    )

    ##########################################################################
    # Training
    ##########################################################################
    parser.add_argument(
        "--opt", type=str,
        default="adam",
        choices=sorted(set(OPTS.keys())),
        help="Type of optimizer to use."
    )
    parser.add_argument(
        "--loss", type=str,
        default="cross_entropy",
        choices=sorted(set(LOSSES.keys())),
        help="Type of optimizer to use."
    )
    parser.add_argument(
        "--lr", type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int,
        default=50,
        help="Epochs"
    )
    parser.add_argument(
        "--batch_size", type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--log_freq", type=int,
        default=8,
        help="How many epochs to wait before logging"
    )

    ##########################################################################
    # Domain Adaptation
    ##########################################################################
    parser.add_argument(
        "--alpha", type=float,
        default=0.25,
        help="What percent of total loss target loss should take up"
    )
    parser.add_argument(
        "--beta", type=float,
        default=0,
        help=(
            "What percent of the target loss classification loss should take up"
        )
    )
    parser.add_argument(
        "--pos_pairs", type=int,
        default=50,
        help="How many positive pairs each target data point should make"
    )
    parser.add_argument(
        "--neg_pairs", type=int,
        default=150,
        help="How many negative pairs each target data point should make"
    )
    parser.add_argument(
        "--da_sessions", type=int,
        default=1,
        help="How many sessions the target set should contain"
    )

    return config_parser, parser


def config_from_args(args, loaded=None):
    config_parser, parser = build_parsers()

    # Load defaults from previous run
    if loaded:
        parser.set_defaults(**vars(loaded))

    # Load defaults from passed configuration file
    parsed_args, _ = config_parser.parse_known_args(args)
    load_defaults(parser, parsed_args.config)

    # Load final values from args
    config = parser.parse_args(args)

    validate_config(config)
    return config
