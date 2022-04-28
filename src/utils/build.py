from multiprocessing import cpu_count

import torch

from src.data import DATASETS
from src.models import MODELS, OPTS, LOSSES


def build_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_datasets(config, device):
    dataset_cls = DATASETS[config.dataset]
    dataset = dataset_cls(config)
    subsets = dataset.split()

    print("Dataset sizes...")
    for s in subsets:
        print(f"  Size: {len(s) if s else 0}")

    loaders = [
        torch.utils.data.DataLoader(
            s,
            shuffle=True,
            batch_size=config.batch_size,
            num_workers=(cpu_count() if device == "cuda" else 0),
            pin_memory=(device == "cuda"),
            collate_fn=dataset.collate_fn
        ) if s else None
        for s in subsets
    ]

    return dataset, loaders


def build_model(config, dataset):
    model_cls = MODELS[config.model.lower()]
    model = model_cls(dataset.input_dim, dataset.target_dim, config)
    return model


def build_optimizer(config, model):
    opt_cls = OPTS[config.opt]
    opt = opt_cls(model.parameters(), lr=config.lr)
    return opt


def build_loss_fn(config):
    loss_cls = LOSSES[config.loss]
    loss_fn = loss_cls()
    return loss_fn


def build(config, device):
    dataset, loaders = build_datasets(config, device)
    model = build_model(config, dataset).to(device)
    opt = build_optimizer(config, model)
    loss_fn = build_loss_fn(config)
    return dataset, loaders, model, opt, loss_fn
