import argparse


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
    if len(config.convs) != len(config.pools):
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
        "--name", type=str,
        default=None
    )
    parser.add_argument(
        "--group", type=str,
        default=None
    )
    parser.add_argument(
        "--notes", type=str,
        default=None
    )

    # Filepaths
    parser.add_argument(
        "--testset", type=str,
        default="data/testsets/testset_spotify.json"
    )
    parser.add_argument(
        "--data", type=str, nargs="+",
        default=["data/datasets/step_data/silent"]
    )
    parser.add_argument(
        "--save_dir", type=str,
        default="data/models/tmp/"
    )

    # Dataset hyperparameters
    parser.add_argument(
        "--max_len", type=int,
        default=1024
    )
    parser.add_argument(
        "--remove_channels", type=int, nargs='*',
        default=[]
    )

    # Model hyperparameters
    parser.add_argument(
        "--fcs", type=int, nargs='+',
        default=[128]
    )
    parser.add_argument(
        "--convs", type=conv_params, nargs='+',
        default=[(3, 1, 16), (3, 1, 32), (3, 1, 64)]
    )
    parser.add_argument(
        "--pools", type=pool_params, nargs='+',
        default=[(2, 2), (2, 2), (2, 2)]
    )
    parser.add_argument(
        "--drop_prob", type=float,
        default=0.25
    )

    # Training hyperparameters
    parser.add_argument(
        "--lr", type=float,
        default=1e-4
    )
    parser.add_argument(
        "--epochs", type=int,
        default=50
    )
    parser.add_argument(
        "--batch_size", type=int,
        default=16
    )

    config = parser.parse_args(args)

    validate_config(config)

    return config
