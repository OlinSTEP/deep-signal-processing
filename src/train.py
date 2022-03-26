import sys

import wandb
import torch
from tqdm import tqdm

from src.config import config_from_args, WANDB_EXCLUDE_KEYS
from src.data import MicClassificationDataset
from src.models.cnn_1d import CNN1D
from src.evaluate import calc_metrics, evaluate


def build_datasets(config, device):
    # TODO: Add to dataset dir and add type selection
    dataset = MicClassificationDataset(config)

    train_idx, dev_idx, test_idx = dataset.build_splits()
    train_set = torch.utils.data.Subset(dataset, train_idx)
    dev_set = torch.utils.data.Subset(dataset, dev_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=(8 if device == "cuda" else 0),
        pin_memory=(device == "cuda"),
        collate_fn=train_set.collate_fn
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory=(device == "cuda"),
        collate_fn=dev_set.collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        batch_size=1,
        num_workers=1,
        pin_memory=(device == "cuda"),
        collate_fn=test_set.collate_fn
    )

    return dataset, train_loader, dev_loader, test_loader


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
    dataset, train_loader, dev_loader, _ = build_datasets(config, device)
    model = build_model(config, dataset).to(device)
    opt = build_optimizer(config, model)
    loss_fn = build_loss_fn(config)

    wandb.init(
        project="STEP Signal Processing",
        entity="step-signal-processing",
        config=config,
        name=config.name,
        group=config.group,
        notes=config.notes,
        config_exclude_keys=WANDB_EXCLUDE_KEYS
    )
    wandb.watch(
        model,
        criterion=loss_fn,
        log_freq=(len(dataset) // config.batch_size) * (config.epochs // 10)
    )

    print("Starting training...")
    train(config, device, train_loader, dev_loader, model, opt, loss_fn)


if __name__ == "__main__":
    main(sys.argv[1:])
