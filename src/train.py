import sys

import wandb
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from src.config import config_from_args, WANDB_EXCLUDE_KEYS
from src.data import SingleFramePhraseDataset
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


def calc_metrics(losses, accuracies, labels=None, preds=None,
                 target_labels=None, prefix=""):
    if prefix:
        prefix = prefix.strip()
        prefix += " "

    metrics = {}
    metrics[prefix + "Loss"] = np.mean(losses)
    metrics[prefix + "Acc"] = np.mean(accuracies)

    if not labels or not preds or not target_labels:
        return metrics

    # To avoid loggign too many metrics, these metrics are not calculated for
    # train data, and therefore do not have a prefix attached to them
    metrics["Confusion Matrix"] = wandb.plot.confusion_matrix(
        y_true=labels,
        preds=preds,
        class_names=target_labels,
        title="Confusion Matrix",
    )

    # sklearn wants idxs as labels and preds are idxs
    target_labels = [i for i in range(len(target_labels))]

    micro_prec, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, preds,
        average="micro", zero_division=0, labels=target_labels
    )
    metrics["Micro Precision"] = micro_prec
    metrics["Micro Recall"] = micro_recall
    metrics["Micro F1"] = micro_f1

    macro_prec, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds,
        average="macro", zero_division=0, labels=target_labels
    )
    metrics["Macro Precision"] = macro_prec
    metrics["Macro Recall"] = macro_recall
    metrics["Macro F1"] = macro_f1

    weighted_prec, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds,
        average="weighted", zero_division=0, labels=target_labels
    )
    metrics["Weighted Precision"] = weighted_prec
    metrics["Weighted Recall"] = weighted_recall
    metrics["Weighted F1"] = weighted_f1

    return metrics


def evaluate(device, dataloader, model, loss_fn):
    # TODO: Move to evaluate.py, add support for testset
    losses = []
    accuracies = []
    all_labels = []
    all_preds = []
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
            all_labels.extend(labels.tolist())
            all_preds.extend(pred.tolist())
    model.train()

    return calc_metrics(
        losses, accuracies,
        labels=all_labels, preds=all_preds,
        target_labels=dataloader.dataset.target_encoder.target_labels,
        prefix="Val"
    )


def train(config, device, train_loader, dev_loader, model, opt, loss_fn):
    model.train()
    for epoch in range(config.epochs):
        losses = []
        accuracies = []
        print(f"\nEpoch {epoch}:")
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

        metrics = {}
        metrics.update(calc_metrics(losses, accuracies, prefix="Train"))
        metrics.update(evaluate(device, dev_loader, model, loss_fn))
        wandb.log(metrics)

        print(
            f"Train Loss: {metrics['Train Loss']:.3f} | "
            f"Train Accuracy: {metrics['Train Acc']:.3f}"
        )
        print(
            f"Dev Loss:   {metrics['Val Loss']:.3f} | "
            f"Dev Accuracy:   {metrics['Val Acc']:.3f}"
        )


def main(args):
    config = config_from_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Processing data...")
    train_set, train_loader, dev_loader = build_datasets(config, device)
    model = build_model(config, train_set).to(device)
    opt = build_optimizer(config, model)
    loss_fn = build_loss_fn(config)

    wandb.init(
        project="EMG Classification",
        entity="step-emg",
        config=config,
        name=config.name,
        group=config.group,
        notes=config.notes,
        config_exclude_keys=WANDB_EXCLUDE_KEYS
    )
    wandb.watch(
        model,
        criterion=loss_fn,
        log_freq=(len(train_set) // config.batch_size) * (config.epochs // 10)
    )

    print("Starting training...")
    train(config, device, train_loader, dev_loader, model, opt, loss_fn)


if __name__ == "__main__":
    main(sys.argv[1:])
