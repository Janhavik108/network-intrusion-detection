"""Builds a small ARFF in the NSL-KDD layout (same attribute names/types) and runs it through."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from arff_to_csv import convert, read_arff  # noqa: E402
from netguard import pipeline  # noqa: E402
from netguard.config import load_config  # noqa: E402

NUMERIC = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "hot", "count", "srv_count",
           "serror_rate", "rerror_rate", "same_srv_rate", "dst_host_count", "dst_host_srv_count"]
BINARY = ["land", "logged_in", "is_guest_login"]


def write_arff(path, n, seed):
    rng = np.random.default_rng(seed)
    lines = ["@relation 'KDDTrain'", ""]
    lines += ["@attribute 'duration' real", "@attribute 'protocol_type' {'tcp','udp', 'icmp'}",
              "@attribute 'service' {'http','ftp','smtp','private'}"]
    lines += [f"@attribute '{c}' real" for c in NUMERIC[1:]]
    lines += [f"@attribute '{c}' {{'0', '1'}}" for c in BINARY]
    lines += ["@attribute 'class' {'normal', 'anomaly'}", "", "@data"]
    for _ in range(n):
        atk = rng.random() < 0.45
        row = [rng.exponential(5), rng.choice(["tcp", "udp", "icmp"]), rng.choice(["http", "ftp", "smtp", "private"])]
        row += [rng.lognormal(3 if atk else 6, 1), rng.lognormal(4, 1), 0, rng.integers(0, 3),
                rng.integers(100, 500) if atk else rng.integers(1, 60), rng.integers(1, 100),
                rng.beta(8, 2) if atk else rng.beta(1, 10), rng.beta(1, 10), rng.beta(2, 4) if atk else rng.beta(6, 2),
                rng.integers(1, 255), rng.integers(1, 255)]
        row += [0, int(rng.random() < 0.5), 0]
        row.append("anomaly" if (atk != (rng.random() < 0.04)) else "normal")
        lines.append(",".join(map(str, row)))
    Path(path).write_text("\n".join(lines))


def test_converter_infers_schema_and_pipeline_runs(tmp_path, monkeypatch):
    train_arff, test_arff = tmp_path / "KDDTrain+.arff", tmp_path / "KDDTest+.arff"
    write_arff(train_arff, 2500, 1)
    write_arff(test_arff, 800, 2)

    df, types = read_arff(train_arff)
    assert "class" in df.columns and len(df) == 2500

    cfg_path = convert(train_arff, test_arff, out_dir=tmp_path, config_out=tmp_path / "cfg.yaml")
    monkeypatch.setenv("NETGUARD_CONFIG", str(cfg_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)

    cfg = load_config()
    assert set(cfg["features"]["categorical"]) == {"protocol_type", "service"}
    assert set(BINARY) <= set(cfg["features"]["numeric"])  # {'0','1'} nominals become numeric
    cfg["paths"] = {"artifacts_dir": str(tmp_path / "A"), "final_model_dir": str(tmp_path / "F"),
                    "logs_dir": str(tmp_path / "L")}
    cfg["training"]["cv_folds"] = 2

    out = pipeline.run(cfg)
    assert out["rows"] == 3300  # 2500 train + 800 fixed test, no random split
    assert out["test_metrics"]["roc_auc"] > 0.8
