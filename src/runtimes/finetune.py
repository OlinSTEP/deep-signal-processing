import sys

from src.utils.setup_all import setup_all
from src.utils.wandb import init_wandb
from .train import train


def main(args):
    config, device, built_objs = setup_all(args)

    dataset, loaders, model, opt, loss_fn = built_objs
    _, dev_target_loader, dev_loader, _, _ = loaders

    init_wandb(config, model=model, loss_fn=loss_fn)

    train(
        config, device,
        dataset, dev_target_loader, dev_loader,
        model, opt, loss_fn
    )


if __name__ == "__main__":
    main(sys.argv[1:])
