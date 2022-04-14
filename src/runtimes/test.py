import sys

import wandb
import torch

from src.config import config_from_args
from src.utils.eval import evaluate
from src.utils.save import load
from src.utils.wandb import init_wandb
from src.utils.build import build_device


def main(args):
    config = config_from_args(args)
    device = build_device()

    if config.load_dir is None:
        raise ValueError("load_dir must be specified for testing")
    config, built_objs = load(args, config.load_dir, device)
    dataset, _, _, test_loader = built_objs[:4]
    model, _, loss_fn = built_objs[4:]

    init_wandb(config, tags=["test"])

    print("Evaluating...")
    metrics = evaluate(device, dataset, test_loader, model, loss_fn)

    # Ints / floats that can use wandb.summary
    summary_metrics = {}
    # Image / graph data that needs wandb.log
    log_metrics = {}

    # Calling wandb.log() with floats / ints adds a "step" which makes graphs
    # with only a single data point. To avoid having these single point graphs
    # show up in our summary view on WandB, we split them out and use summary

    for metric, value in metrics.items():
        if isinstance(value, int) or isinstance(value, float):
            summary_metrics[metric] = value
        else:
            log_metrics[metric] = value

    wandb.summary.update(summary_metrics)
    wandb.log(log_metrics)

    for metric, value in summary_metrics.items():
        print(f"Test {metric}: {value:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
