import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper())
    fmt = log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = log_cfg.get("file", "logs/run.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handlers = [logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    logger = logging.getLogger(config.get("project", {}).get("name", "capstone"))
    return logger


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
