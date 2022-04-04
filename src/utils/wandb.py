import wandb

from src.config import WANDB_EXCLUDE_KEYS


def init_wandb(config, tags=None):
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