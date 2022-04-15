import wandb
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import torch


def calc_metrics(
    losses, accuracies,
    labels=None, preds=None, target_labels=None, prefix="", use_wandb=True
):
    if prefix:
        prefix = prefix.strip()
        prefix += " "

    metrics = {}
    metrics[prefix + "Loss"] = np.mean(losses)
    metrics[prefix + "Acc"] = np.mean(accuracies)

    if not labels or not preds or not target_labels:
        return metrics

    if use_wandb:
        # To avoid logging too many metrics, these metrics are not calculated for
        # train data, and therefore do not have a prefix attached to them
        metrics["Confusion Matrix"] = wandb.plot.confusion_matrix(
            y_true=labels,
            preds=preds,
            class_names=target_labels,
            title="Confusion Matrix",
        )

    # sklearn wants idxs as labels and preds are idxs
    target_labels = [i for i in range(len(target_labels))]

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


def evaluate(device, dataset, dataloader, model, loss_fn, prefix=""):
    losses = []
    accuracies = []
    all_labels = []
    all_preds = []
    # model.eval()
    with torch.no_grad():
        for datapoint in dataloader:
            inputs = datapoint["input"].to(device)
            labels = datapoint["target"].to(device)
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
        target_labels=dataset.target_encoder.target_labels,
        prefix=prefix
    )
