import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class NetGuardError(Exception):
    """Base error for pipeline failures."""


class DataValidationError(NetGuardError):
    pass


def load_config(path=None) -> dict:
    path = Path(path or os.getenv("NETGUARD_CONFIG", ROOT / "config" / "config.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve(cfg: dict, relative: str) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def get_logger(name: str, cfg: dict | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    logs_dir = resolve(cfg, cfg["paths"]["logs_dir"]) if cfg else ROOT / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logs_dir / "netguard.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass
    return logger
