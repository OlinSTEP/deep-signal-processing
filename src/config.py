import argparse


def config_from_args(args):
    parser = argparse.ArgumentParser("Train EMG Model")

    # Dataset filepaths
    parser.add_argument(
        "--testset", type=str,
        default="data/testsets/testset_spotify.json"
    )
    parser.add_argument(
        "--data", type=str,
        default="data/step_data/silent/"
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

    config = parser.parse_args(args)
    return config
