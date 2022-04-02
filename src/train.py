import sys
from multiprocessing import cpu_count

import wandb
import torch
from tqdm import tqdm


from .config import config_from_args, WANDB_EXCLUDE_KEYS
from .data import DATASETS
from .models import MODELS, OPTS, LOSSES
from .evaluate import calc_metrics, evaluate


def build_datasets(config, device):
    dataset_cls = DATASETS[config.dataset]
    dataset = dataset_cls(config)
    train_set, dev_set, test_set = dataset.split()

    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=(cpu_count() if device == "cuda" else 0),
        pin_memory=(device == "cuda"),
        collate_fn=dataset.collate_fn
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=(cpu_count() if device == "cuda" else 0),
        pin_memory=(device == "cuda"),
        collate_fn=dataset.collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=(cpu_count() if device == "cuda" else 0),
        pin_memory=(device == "cuda"),
        collate_fn=dataset.collate_fn
    )

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


def train(
    config, device,
    dataset, train_loader, dev_loader,
    model, opt, loss_fn
):
    print("Starting training...")
    model.train()
    for epoch in range(config.epochs):
        losses = []
        accuracies = []
        print(f"\nEpoch {epoch}:")
        for datapoint in tqdm(train_loader):
            inputs = datapoint["input"].to(device)
            labels = datapoint["target"].to(device)

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
        if epoch % config.log_freq == 0 or epoch + 1 == config.epochs:
            metrics.update(evaluate(device, dataset, dev_loader, model, loss_fn))
        wandb.log(metrics)

        print(
            f"Train Loss: {metrics['Train Loss']:.3f} | "
            f"Train Accuracy: {metrics['Train Acc']:.3f}"
        )
        if epoch % config.log_freq == 0 or epoch + 1 == config.epochs:
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
        project=config.project,
        entity="step-emg",
        config=config,
        name=config.name,
        group=config.group,
        notes=config.notes,
        config_exclude_keys=WANDB_EXCLUDE_KEYS,
        mode=("disabled" if config.wandb_off else "online")
    )
    wandb.watch(
        model,
        criterion=loss_fn,
        log_freq=(len(dataset) // config.batch_size) * (config.epochs // 10)
    )

    train(
        config, device,
        dataset, train_loader, dev_loader,
        model, opt, loss_fn
    )


if __name__ == "__main__":
    main(sys.argv[1:])
