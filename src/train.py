import sys

import torch
import numpy as np
from tqdm import tqdm

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


def evaluate(device, dataloader, model, loss_fn):
    # TODO: Move to evaluate.py, add support for testset
    losses = []
    accuracies = []
    model.eval()
    with torch.no_grad():
        for datapoint in dataloader:
            inputs = datapoint["emg"].to(device)
            labels = datapoint["text"].to(device)
            out = model(inputs)
            loss = loss_fn(out, labels)

            _, pred = torch.max(out, dim=1)
            acc = torch.sum(pred == labels) / labels.size()[0]
            accuracies.append(acc.item())
            losses.append(loss.item())
    model.train()

    total_loss = np.mean(losses)
    total_acc = np.mean(accuracies)

    return total_loss, total_acc


def train(config, device, train_loader, dev_loader, model, opt, loss_fn):
    model.train()
    for _ in range(config.epochs):
        losses = []
        accuracies = []
        for datapoint in tqdm(train_loader):
            inputs = datapoint["emg"].to(device)
            labels = datapoint["text"].to(device)

            opt.zero_grad()

            out = model(inputs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()

            _, pred = torch.max(out, dim=1)
            acc = torch.sum(pred == labels) / labels.size()[0]
            accuracies.append(acc.item())
            losses.append(loss.item())
        model.save("model", config.save_dir)

        total_loss = np.mean(losses)
        total_acc = np.mean(accuracies)
        val_loss, val_acc = evaluate(device, dev_loader, model, loss_fn)
        print(f"Train Loss: {total_loss:.3f} | Train Accuracy: {total_acc:.3f}")
        print(f"Dev Loss:   {val_loss:.3f} | Dev Accuracy:   {val_acc:.3f}")


def main(args):
    config = config_from_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, train_loader, dev_loader = build_datasets(config)
    model = build_model(config, train_set).to(device)
    opt = build_optimizer(config, model)
    loss_fn = build_loss_fn(config)

    train(config, device, train_loader, dev_loader, model, opt, loss_fn)


if __name__ == "__main__":
    main(sys.argv[1:])
