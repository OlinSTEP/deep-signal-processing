import argparse


def config_from_args(args):
    parser = argparse.ArgumentParser("Train EMG Model")

    parser.add_argument("--max_len", type=int, default=1024)

    config = parser.parse_args(args)
    return config
