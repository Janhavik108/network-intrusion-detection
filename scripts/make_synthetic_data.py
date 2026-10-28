"""Generate a SYNTHETIC flow-record dataset for smoke-testing the pipeline.

This is not real traffic. Any metric computed on it says nothing about real-world
intrusion detection quality. Swap in NSL-KDD / UNSW-NB15 / CIC-IDS for real results.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    attack = rng.random(n) < 0.35
    kind = np.where(attack, rng.choice(["dos", "probe", "r2l"], n, p=[0.5, 0.35, 0.15]), "normal")
    is_dos, is_probe = kind == "dos", kind == "probe"

    df = pd.DataFrame({
        "duration": np.where(is_probe, rng.exponential(0.5, n), rng.exponential(30, n)),
        "src_bytes": np.where(is_dos, rng.lognormal(3, 1, n), rng.lognormal(6.5, 1.4, n)),
        "dst_bytes": np.where(attack, rng.lognormal(3.5, 1.5, n), rng.lognormal(7, 1.5, n)),
        "count": np.where(is_dos, rng.integers(150, 511, n), rng.integers(1, 80, n)),
        "srv_count": np.where(is_probe, rng.integers(1, 15, n), rng.integers(1, 120, n)),
        "serror_rate": np.clip(np.where(is_dos, rng.beta(8, 2, n), rng.beta(1, 12, n)), 0, 1),
        "rerror_rate": np.clip(np.where(is_probe, rng.beta(5, 3, n), rng.beta(1, 14, n)), 0, 1),
        "same_srv_rate": np.clip(np.where(is_probe, rng.beta(2, 5, n), rng.beta(6, 1.5, n)), 0, 1),
        "logged_in": np.where(kind == "r2l", rng.random(n) < 0.7, rng.random(n) < 0.55).astype(int),
        "protocol_type": rng.choice(["tcp", "udp", "icmp"], n, p=[0.7, 0.2, 0.1]),
        "service": rng.choice(["http", "ftp", "smtp", "ssh", "dns", "private"], n),
    })
    # label noise so no model scores a perfect 1.0
    flip = rng.random(n) < 0.03
    attack = np.where(flip, ~attack, attack)
    df["label"] = np.where(attack, "attack", "normal")
    # sprinkle missing values to exercise the imputers
    for col in ("duration", "src_bytes", "service"):
        df.loc[rng.random(n) < 0.01, col] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/raw/traffic.csv")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make(a.rows, a.seed).to_csv(out, index=False)
    print(f"Wrote {a.rows} synthetic rows to {out}")
