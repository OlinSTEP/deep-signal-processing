from src.config import config_from_args
from src.utils.save import load
from src.utils.build import build_device, build


def setup_all(args):
    config = config_from_args(args)
    device = build_device()
    if config.load_dir:
        config, built_objs = load(args, config.load_dir, device)
    else:
        built_objs = build(config, device)
    return config, device, built_objs
