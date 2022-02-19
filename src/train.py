import sys

import torch

from src.config import config_from_args
from src.dataset import SingleFramePhraseDataset
from src.models.cnn_1d import CNN1D


def build_datasets(config, device):
    # TODO: Add to dataset dir and add type selection
    train_set = SingleFramePhraseDataset(config, dev=False, test=False)
    dev_set = SingleFramePhraseDataset(config, dev=True, test=False)
    dev_set.set_encoding(train_set)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=(8 if device=="cuda" else 0),
        pin_memory=(device=="cuda"),
        collate_fn=train_set.collate_fn
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory=(device=="cuda"),
        collate_fn=dev_set.collate_fn
    )

    return train_set, train_loader, dev_loader


def build_model(config, dataset):
    # TODO: Add to model dir and add type selection
    model = CNN1D(dataset.input_dim, dataset.target_dim, config)
    return model


def build_optimizer(config, model):
    # TODO: Add to new dir and add type selection
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    return opt


def build_loss_fn(config):
    # TODO: Add to new dir and add type selection
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn


def main(args):
    config = config_from_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, train_loader, dev_loader = build_datasets(config)
    model = build_model(config, train_set).to(device)
    opt = build_optimizer(config, model)
    loss_fn = build_loss_fn(config)

    train(train_loader, dev_loader, model, opt, loss_fn)


def train(train_loader, dev_loader, model, opt, loss_fn):
    pass


if __name__ == "__main__":
    main(sys.argv[1:])
