import os
import pickle
import argparse

from src.config import build_parsers, WANDB_EXCLUDE_KEYS
from src.utils.build import build


def save(config, model, name="model"):
    save_dir = os.path.join(config.save_dir, name)

    model.save(name, save_dir)
    config_save_path = os.path.join(save_dir, "config.pkl")

    # Copy the config without WandB options
    config_contents = {
        key: value
        for key, value in vars(config).items()
        if key not in WANDB_EXCLUDE_KEYS
    }
    config = argparse.Namespace(**config_contents)
    with open(config_save_path, "wb") as f:
        pickle.dump(config, f)
    print(f"Config saved to {config_save_path}")


def load(args, load_dir, device):
    config_load_path = os.path.join(load_dir, "config.pkl")
    with open(config_load_path, "rb") as f:
        loaded_config = pickle.load(f)
    print(f"Config loaded from {config_load_path}")

    # Overwrite with any new command line options
    _, parser = build_parsers()
    config = parser.parse_args(args, namespace=loaded_config)

    build_objs = build(config, device)
    dataset, train_loader, dev_loader, test_loader = build_objs[:4]
    model, opt, loss_fn = build_objs[4:]

    model_load_path = os.path.join(load_dir, "model.h5")
    model.load(model_load_path, device)

    build_objs = (
        dataset, train_loader, dev_loader, test_loader,
        model, opt, loss_fn
    )
    return config, build_objs
