import os
import sys
import json

import torch
import scipy
import numpy as np

from src.config import config_from_args
from src.utils.eval import calc_metrics
from src.utils.save import load
from src.utils.build import build_device


def save_idx(loader, idx, save_dir):
    data, target, _ = loader.load(idx)
    reg_sr = data[0][0]
    reg_data = np.array([data[0][1], data[1][1]], dtype=np.int16)
    throat_sr = data[2][0]
    throat_data = np.array([data[2][1], data[3][1]], dtype=np.int16)

    print(f"Writing {os.path.join(save_dir, str(idx))}...")

    file_path = os.path.join(save_dir, f"{idx}_info.json")
    save_dict = {
        "target": target
    }
    with open(file_path, "w") as f:
        json.dump(save_dict, f)
    print(f"  Wrote {file_path}")

    file_path = os.path.join(save_dir, f"{idx}_reg_audio.wav")
    scipy.io.wavfile.write(
        file_path,
        reg_sr,
        np.transpose(reg_data)
    )
    print(f"  Wrote {file_path}")

    file_path = os.path.join(save_dir, f"{idx}_throat_audio.wav")
    scipy.io.wavfile.write(
        file_path,
        throat_sr,
        np.transpose(throat_data)
    )
    print(f"  Wrote {file_path}")


def main(args):
    config = config_from_args(args)
    device = build_device()
    os.makedirs(config.save_dir, exist_ok=True)

    if config.load_dir is None:
        raise ValueError("load_dir must be specified")
    config, built_objs = load(args, config.load_dir, device)
    dataset, *_ = built_objs[:4]
    model, _, loss_fn = built_objs[4:]

    loader = dataset.loader
    *_, test_idxs = loader.build_splits()
    print(len(test_idxs))

    losses = []
    accuracies = []
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for idx in test_idxs:
            datapoint = dataset[idx]
            inputs = datapoint["input"][None, :].to(device)
            label = datapoint["target"]
            out = model(inputs)
            loss = loss_fn(out, torch.tensor([label]))

            _, pred = torch.max(out[0, :], dim=0)
            is_correct = torch.sum(pred == label).item()
            accuracies.append(is_correct)
            losses.append(loss.item())
            all_labels.append(label)
            all_preds.append(pred.tolist())

            if is_correct == 0:
                save_idx(loader, idx, config.save_dir)

    metrics = calc_metrics(
        losses, accuracies,
        labels=all_labels, preds=all_preds,
        target_labels=dataset.target_encoder.target_labels,
        use_wandb=False
    )

    for metric, value in metrics.items():
        print(f"Test {metric}: {value:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
