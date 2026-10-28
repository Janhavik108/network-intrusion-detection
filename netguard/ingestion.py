"""Stage 1: pull raw records from MongoDB (or a CSV fallback) and split them."""
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import NetGuardError, get_logger, resolve

log = get_logger("netguard.ingestion")


def _load_from_mongo(cfg: dict) -> pd.DataFrame:
    from pymongo import MongoClient

    client = MongoClient(os.environ["MONGO_URI"])
    coll = client[cfg["mongo"]["database"]][cfg["mongo"]["collection"]]
    df = pd.DataFrame(list(coll.find({}, {"_id": 0})))
    if df.empty:
        raise NetGuardError(
            f"MongoDB collection {cfg['mongo']['database']}.{cfg['mongo']['collection']} is empty. "
            "Run scripts/push_to_mongo.py first."
        )
    return df


def _load_from_csv(cfg: dict, csv_path: str | None) -> pd.DataFrame:
    path = resolve(cfg, csv_path or cfg["data"]["csv_path"])
    if not path.exists():
        raise NetGuardError(f"CSV not found at {path}. Add a dataset or run scripts/make_synthetic_data.py.")
    return pd.read_csv(path)


def load_raw(cfg: dict, csv_path: str | None = None) -> tuple[pd.DataFrame, str]:
    if os.getenv("MONGO_URI") and not csv_path:
        log.info("Reading records from MongoDB")
        return _load_from_mongo(cfg), "mongodb"
    log.info("MONGO_URI not set (or csv override given), reading CSV")
    return _load_from_csv(cfg, csv_path), "csv"


def to_binary_target(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Map the raw label column to target = 1 for attack, 0 for benign."""
    col = cfg["data"]["target"]
    if col not in df.columns:
        raise NetGuardError(f"Target column '{col}' missing from data")
    benign = {str(b).strip().lower() for b in cfg["data"]["benign_labels"]}
    df = df.copy()
    df["target"] = (~df[col].astype(str).str.strip().str.lower().isin(benign)).astype(int)
    return df.drop(columns=[col])


def ingest(cfg: dict, run_dir: Path, csv_path: str | None = None) -> dict:
    df, source = load_raw(cfg, csv_path)
    df = to_binary_target(df, cfg)
    fixed_test = cfg["data"].get("test_csv_path")
    if fixed_test:
        # Official benchmark split (e.g. NSL-KDD KDDTest+): train on everything loaded, test on the file.
        train = df
        test = to_binary_target(_load_from_csv(cfg, fixed_test), cfg)
        log.info("Using fixed test file %s", fixed_test)
    else:
        train, test = train_test_split(
            df,
            test_size=cfg["data"]["test_size"],
            random_state=cfg["data"]["seed"],
            stratify=df["target"],
        )
    out = run_dir / "ingested"
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    total = len(train) + len(test)
    log.info("Ingested %d rows from %s (train=%d, test=%d, train attack rate=%.3f)",
             total, source, len(train), len(test), train["target"].mean())
    return {"source": source, "rows": total, "train_path": str(out / "train.csv"),
            "test_path": str(out / "test.csv"), "attack_rate": float(train["target"].mean())}
