from typing import List, Tuple, Optional, Callable, Any
from argparse import Namespace

from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader

from src.data import DATASETS, Dataset
from src.models import MODELS, OPTS, LOSSES, Model


def build_device() -> str:
    """
    Builds device to run off

    :rtype str: Device string
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_datasets(config: Namespace, device: str) -> \
        Tuple[Dataset, List[Optional[DataLoader]]]:
    """
    Builds dataset and dataloaders

    :param config Namespace: Config to use for building
    :param device str: Device to use for building
    :rtype Tuple[Dataset, List[torch.utils.DataLoader]]: Entire dataset, then a
    list of data loaders. One data loader is created for each dataset split.
    Data loaders are None if split has size 0
    """
    dataset_cls = DATASETS[config.dataset]
    dataset = dataset_cls(config)
    subsets = dataset.split()

    print("Dataset sizes...")
    for s in subsets:
        print(f"  Size: {len(s) if s else 0}")

    loaders = [
        DataLoader(
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


def build_model(config: Namespace, dataset: Dataset) -> Model:
    """
    Builds model

    :param config Namespace: Config to use for building
    :param dataset Dataset: Dataset to pull input and target dims from
    :rtype Model: Built model
    """
    model_cls = MODELS[config.model.lower()]
    model = model_cls(dataset.input_dim, dataset.target_dim, config)
    return model


def build_optimizer(config: Namespace, model: Model) -> torch.optim.Optimizer:
    """
    Builds optimizer

    :param config Namespace: Config to use for building
    :param model Model: Model to build optimizer for
    :rtype torch.optim.Optimizer: Built optimizer
    """
    opt_cls = OPTS[config.opt]
    opt = opt_cls(model.parameters(), lr=config.lr)
    return opt


def build_loss_fn(config: Namespace) -> Callable[..., Any]:
    """
    Builds loss function

    :param config Namespace: Config to use for building
    :rtype Callable[..., Any]: Built torch loss function
    """
    loss_cls = LOSSES[config.loss]
    loss_fn = loss_cls()
    return loss_fn


def build(config: Namespace, device: str) -> Tuple[
    Dataset,
    List[Optional[DataLoader]],
    Model,
    torch.optim.Optimizer,
    Callable[..., Any]
]:
    """
    Convenience function to build all objects for training

    :param config Namespace: Config to build with
    :param device str: Device string
    :rtype Tuple[
        Dataset,
        List[Optional[DataLoader]],
        Model,
        torch.optim.Optimizer,
        Callable[..., Any]
    ]: All built objects, see build_datasets, build_model, build_optimizer and
    build_loss_fn respectively
    """
    dataset, loaders = build_datasets(config, device)
    model = build_model(config, dataset).to(device)
    opt = build_optimizer(config, model)
    loss_fn = build_loss_fn(config)
    return dataset, loaders, model, opt, loss_fn
