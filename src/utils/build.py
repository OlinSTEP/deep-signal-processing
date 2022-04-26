from multiprocessing import cpu_count

import torch

from src.data import DATASETS
from src.models import MODELS, OPTS, LOSSES


def build_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_datasets(config, device):
    dataset_cls = DATASETS[config.dataset]
    dataset = dataset_cls(config)
    train_set, dev_set, test_set = dataset.split()

    print(f"Train set size: {len(train_set if train_set else 0)}")
    print(f"Dev set size: {len(dev_set) if dev_set else 0}")
    print(f"Test set size: {len(test_set) if test_set else 0}")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=(cpu_count() if device == "cuda" else 0),
        pin_memory=(device == "cuda"),
        collate_fn=dataset.collate_fn
    ) if train_set else None
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=(cpu_count() if device == "cuda" else 0),
        pin_memory=(device == "cuda"),
        collate_fn=dataset.collate_fn
    ) if dev_set else None
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=(cpu_count() if device == "cuda" else 0),
        pin_memory=(device == "cuda"),
        collate_fn=dataset.collate_fn
    ) if test_set else None

    return dataset, train_loader, dev_loader, test_loader


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
    dataset, train_loader, dev_loader, test_loader = build_datasets(config, device)
    model = build_model(config, dataset).to(device)
    opt = build_optimizer(config, model)
    loss_fn = build_loss_fn(config)
    return (
        dataset, train_loader, dev_loader, test_loader,
        model, opt, loss_fn
    )
