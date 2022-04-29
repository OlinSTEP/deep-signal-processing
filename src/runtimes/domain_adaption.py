import sys

import wandb
import torch
from tqdm import tqdm
import numpy as np

from src.data.domain_adaption_dataset import DomainAdaptionDataset
from src.utils.eval import calc_metrics, evaluate
from src.utils.setup_all import setup_all
from src.utils.save import save
from src.utils.wandb import init_wandb


def contrastive_loss(x, y, class_eq, margin=1):
    dist = torch.nn.functional.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()

def domain_adaption(
    config, device,
    dataset, train_loader, dev_loader,
    model, opt, loss_fn, da_loss_fn
):
    print("Starting training...")
    top_acc = 0
    model.train()
    for epoch in range(config.epochs):
        accuracies = []
        classif_losses, cont_losses, losses = [], [], []
        print(f"\nEpoch {epoch}:")
        for datapoint in tqdm(train_loader):
            source_inputs = datapoint["source_input"].to(device)
            source_targets = datapoint["source_target"].to(device)
            target_inputs = datapoint["target_input"].to(device)
            is_pos = datapoint["is_pos"].to(device)

            source_out, source_feats = model(source_inputs, features=True)
            _, target_feats = model(target_inputs, features=True)

            classif_loss = loss_fn(source_out, source_targets)
            cont_loss = da_loss_fn(source_feats, target_feats, is_pos)
            loss = classif_loss * (1 - config.alpha) + cont_loss * config.alpha

            opt.zero_grad()
            loss.backward()
            opt.step()

            _, pred = torch.max(source_out, dim=1)
            acc = torch.sum(pred == source_targets) / source_targets.size()[0]
            accuracies.append(acc.item())
            classif_losses.append(classif_loss.item())
            cont_losses.append(cont_loss.item())
            losses.append(loss.item())

        metrics = {}
        metrics.update(calc_metrics(losses, accuracies, prefix="Train"))
        metrics.update({
            "Train Classif Loss": np.mean(classif_losses),
            "Train Cont Loss": np.mean(cont_losses),
        })
        if (
            (epoch % config.log_freq == 0 or epoch + 1 == config.epochs)
            and dev_loader
        ):
            metrics.update(evaluate(
                device, dataset, dev_loader, model, loss_fn, prefix="Val"
            ))
            if metrics["Val Acc"] > top_acc:
                top_acc = metrics["Val Acc"]
                save(config, model)
        wandb.log(metrics)

        print(
            f"Train Classif Loss: {metrics['Train Classif Loss']:.3f} | "
            f"Train Cont Loss: {metrics['Train Cont Loss']:.3f}"
        )
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
    train_loader, _, dev_loader, _ = loaders

    if not issubclass(type(dataset), DomainAdaptionDataset):
        raise ValueError(
            "Subclass of DomainAdaptionDataset must be used for domain "
            "adaption"
        )

    init_wandb(config, model=model, loss_fn=loss_fn)

    domain_adaption(
        config, device,
        dataset, train_loader, dev_loader,
        model, opt, loss_fn, contrastive_loss
    )


if __name__ == "__main__":
    main(sys.argv[1:])
