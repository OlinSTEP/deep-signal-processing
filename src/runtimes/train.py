import sys

import wandb
import torch
from tqdm import tqdm

from src.config import config_from_args
from src.utils.eval import calc_metrics, evaluate
from src.utils.build import build
from src.utils.save import save, load
from src.utils.wandb import init_wandb


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
        save(config, model)

        metrics = {}
        metrics.update(calc_metrics(
            losses, accuracies, prefix="Train"
        ))
        if epoch % config.log_freq == 0 or epoch + 1 == config.epochs:
            metrics.update(evaluate(
                device, dataset, dev_loader, model, loss_fn, prefix="Val"
            ))
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

    if config.load_dir:
        config, built_objs = load(args, config.load_dir, device)
    else:
        built_objs = build(config, device)
    dataset, train_loader, dev_loader, _ = built_objs[:4]
    model, opt, loss_fn = built_objs[4:]

    init_wandb(config)
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
