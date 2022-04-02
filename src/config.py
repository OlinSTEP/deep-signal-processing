import argparse

from .models import MODELS, OPTS, LOSSES
from .data import DATASETS


WANDB_EXCLUDE_KEYS = ["name", "group", "notes"]


# PARSING TYPES

def conv_params(s):
    try:
        kernel_len, kernel_stride, out_size = map(int, s.split(''))
        return kernel_len, kernel_stride, out_size
    except:
        raise argparse.ArgumentError(
            "Argument must be of form kernel_len,kernel_stride,out_size"
        )


def pool_params(s):
    try:
        pool_len, pool_stride = map(int, s.split(''))
        return pool_len, pool_stride
    except:
        raise argparse.ArgumentError(
            "Argument must be of form pool_len,pool_stride"
        )


# VALIDATION FUNCS

def validate_convs_pools(config):
    if config.pools is not None and len(config.convs) != len(config.pools):
        raise argparse.ArgumentError(
            "Equal number of conv layers and pooling layers must be defined. "
            f"{len(config.convs)} conv layers and"
            f"{len(config.pools)} pooling layers passed."
        )


def validate_config(config):
    validate_convs_pools(config)


def config_from_args(args):
    parser = argparse.ArgumentParser("Train Signal Processing Model")

    # WandB
    parser.add_argument(
        "---project", type=str,
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

    # Filepaths
    parser.add_argument(
        "--data", type=str,
        default="data/processed_data/SpringBreakAudio",
        help="Dataset to load. Should contain session dirs within"
    )
    parser.add_argument(
        "--save_dir", type=str,
        default="data/models/tmp/",
        help="Directory to save model to"
    )

    # Dataset hyperparameters
    parser.add_argument(
        "--dataset", type=str,
        default="throat_mic_classif",
        help=(
            "Type of dataset to use. Options: "
            f"{', '.join(DATASETS.keys())}"
        )
    )

    ## Splits
    parser.add_argument(
        "--stratify", type=int,
        default=1,
        help=(
            "Whether to use stratified sampling for train / dev / test split."
            "Guarentees even equal distribution of labels between splits, but"
            " breaks for small datasets. 1 for True, 0 for False"
        )
    )

    ## Augmentation
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
        "--aug_spec", type=int,
        default=0,
        help="Whether to use randomly mask spectogram. 1 for True, 0 for False"
    )

    ## Length
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

    ## Audio Specific
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

    # Model hyperparameters
    parser.add_argument(
        "--model", type=str,
        default="2d_cnn",
        help=(
            "Type of model to train. Options: "
            f"{', '.join(MODELS.keys())}"
        )
    )

    ## Layers
    parser.add_argument(
        "--fcs", type=int, nargs='+',
        default=[128],
        help="Fully connected layers. See any model docstring for format"
    )
    parser.add_argument(
        "--convs", type=conv_params, nargs='+',
        default=[(5, 2, 8), (3, 2, 16), (3, 2, 32), (3, 2, 64)],
        help="CNN layers. See CNN[1|2]D docstring for format"
    )
    parser.add_argument(
        "--pools", type=pool_params, nargs='+',
        default=None,
        help="Pooling layers. See CNN[1|2]D docstring for format"
    )

    ## Regularization
    parser.add_argument(
        "--drop_prob", type=float,
        default=0.25,
        help="Dropout probability."
    )

    # Training
    parser.add_argument(
        "--opt", type=str,
        default="adam",
        help=(
            "Type of optimizer to use. Options: "
            f"{', '.join(OPTS.keys())}"
        )
    )
    parser.add_argument(
        "--loss", type=str,
        default="cross_entropy",
        help=(
            "Type of optimizer to use. Options: "
            f"{', '.join(LOSSES.keys())}"
        )
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

    config = parser.parse_args(args)

    validate_config(config)

    return config
