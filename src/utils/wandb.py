import os
import wandb

from src.config import WANDB_EXCLUDE_KEYS


def init_wandb(config, model=None, loss_fn=None, tags=None):
    wandb.init(
        project=config.project,
        entity="step-emg",
        config=config,
        name=config.name,
        group=config.group,
        notes=config.notes,
        tags=tags,
        config_exclude_keys=WANDB_EXCLUDE_KEYS,
        mode=("disabled" if config.wandb_off else "online")
    )

    if config.save_dir == "auto":
        config.save_dir = os.path.join("data/models", wandb.run.name)

    if model is not None:
        wandb.watch(
            model,
            criterion=loss_fn,
            log_freq=config.log_freq
        )
