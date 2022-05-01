import sys

import wandb
import torch
from tqdm import tqdm

from src.utils.setup_all import setup_all
from src.utils.wandb import init_wandb
from src.utils.eval import calc_metrics, evaluate
from src.utils.save import save


def finetune(
    config, device,
    dataset, train_loader, dev_train_loader, dev_loader,
    model, opt, loss_fn
):
    print("Starting training...")
    top_acc = 0
    model.train()
    for epoch in range(config.epochs):
        losses = []
        accuracies = []
        print(f"\nEpoch {epoch}:")
        for datapoint in tqdm(dev_train_loader):
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
            losses, accuracies, prefix="Dev Train",
        ))
        if epoch % config.log_freq == 0 or epoch + 1 == config.epochs:
            # Only want to see loss / acc on the train set
            metrics.update(evaluate(
                device, dataset, train_loader, model, loss_fn,
                prefix="Train", all_metrics=False
            ))
            metrics.update(evaluate(
                device, dataset, dev_loader, model, loss_fn,
                prefix="Val", all_metrics=True
            ))
            if metrics["Val Acc"] > top_acc:
                top_acc = metrics["Val Acc"]
                save(config, model)
        wandb.log(metrics)

        print(
            f"Dev Train Loss: {metrics['Dev Train Loss']:.3f} | "
            f"Dev Train Accuracy: {metrics['Dev Train Acc']:.3f}"
        )
        if epoch % config.log_freq == 0 or epoch + 1 == config.epochs:
            print(
                f"Train Loss: {metrics['Train Loss']:.3f} | "
                f"Train Accuracy: {metrics['Train Acc']:.3f}"
            )
            print(
                f"Dev Loss:   {metrics['Val Loss']:.3f} | "
                f"Dev Accuracy:   {metrics['Val Acc']:.3f}"
            )
    print(f"Top Validation Accuracy: {top_acc}")
    wandb.summary["Top Accuracy"] = top_acc


def main(args):
    config, device, built_objs = setup_all(args)

    dataset, loaders, model, opt, loss_fn = built_objs
    train_loader, dev_target_loader, dev_loader, _, _ = loaders

    init_wandb(config, model=model, loss_fn=loss_fn)

    finetune(
        config, device,
        dataset, train_loader, dev_target_loader, dev_loader,
        model, opt, loss_fn
    )


if __name__ == "__main__":
    main(sys.argv[1:])
