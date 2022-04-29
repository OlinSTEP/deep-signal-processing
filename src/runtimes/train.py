import sys

import wandb
import torch
from tqdm import tqdm

from src.utils.eval import calc_metrics, evaluate
from src.utils.setup_all import setup_all
from src.utils.save import save
from src.utils.wandb import init_wandb


def train(
    config, device,
    dataset, train_loader, dev_loader,
    model, opt, loss_fn
):
    print("Starting training...")
    top_acc = 0
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

        metrics = {}
        metrics.update(calc_metrics(
            losses, accuracies, prefix="Train"
        ))
        if epoch % config.log_freq == 0 or epoch + 1 == config.epochs:
            metrics.update(evaluate(
                device, dataset, dev_loader, model, loss_fn, prefix="Val"
            ))
            if metrics["Val Acc"] > top_acc:
                top_acc = metrics["Val Acc"]
                save(config, model)
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
    print(f"Top Validation Accuracy: {top_acc}")
    wandb.summary["Top Accuracy"] = top_acc


def main(args):
    config, device, built_objs = setup_all(args)

    dataset, loaders, model, opt, loss_fn = built_objs
    train_loader, dev_loader, _ = loaders

    init_wandb(config, model=model, loss_fn=loss_fn)

    train(
        config, device,
        dataset, train_loader, dev_loader,
        model, opt, loss_fn
    )


if __name__ == "__main__":
    main(sys.argv[1:])
