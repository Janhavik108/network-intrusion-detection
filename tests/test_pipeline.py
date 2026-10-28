import json
import sys
from pathlib import Path

import boto3
import mongomock
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from moto import mock_aws

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_data import make  # noqa: E402
from netguard import ingestion, pipeline, storage, validation  # noqa: E402
from netguard.config import DataValidationError, load_config  # noqa: E402


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    base = load_config()
    base["paths"] = {"artifacts_dir": str(tmp_path / "Artifacts"),
                     "final_model_dir": str(tmp_path / "final_model"),
                     "logs_dir": str(tmp_path / "logs")}
    base["training"]["cv_folds"] = 2
    csv = tmp_path / "traffic.csv"
    make(3000, 1).to_csv(csv, index=False)
    base["data"]["csv_path"] = str(csv)
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    return base


def test_full_pipeline_trains_five_models_and_promotes(cfg):
    out = pipeline.run(cfg)
    meta = json.loads((Path(cfg["paths"]["final_model_dir"]) / "metadata.json").read_text())
    assert set(meta["all_models"]) == {"random_forest", "decision_tree", "gradient_boosting",
                                       "logistic_regression", "adaboost"}
    assert out["promoted"] is True
    assert out["test_metrics"]["roc_auc"] > 0.8


def test_champion_is_kept_when_candidate_is_worse(cfg):
    pipeline.run(cfg)
    meta_path = Path(cfg["paths"]["final_model_dir"]) / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["test_metrics"]["f1"] = 0.9999  # pretend the champion is nearly perfect
    meta_path.write_text(json.dumps(meta))
    assert pipeline.run(cfg)["promoted"] is False


def test_validation_rejects_missing_column(cfg, tmp_path):
    df = make(500, 2).drop(columns=["src_bytes"])
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(Exception, match="src_bytes"):
        pipeline.run(cfg, csv_path=str(bad))


def test_validation_flags_single_class(cfg, tmp_path):
    df = make(500, 3)
    df["label"] = "normal"
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    df = ingestion.to_binary_target(df, cfg)
    df.to_csv(tmp_path / "a.csv", index=False)
    with pytest.raises(DataValidationError, match="one class"):
        validation.validate(cfg, str(tmp_path / "a.csv"), str(tmp_path / "a.csv"), run_dir)


def test_ingestion_reads_from_mongodb(cfg, tmp_path, monkeypatch):
    client = mongomock.MongoClient()
    client[cfg["mongo"]["database"]][cfg["mongo"]["collection"]].insert_many(
        make(400, 4).to_dict("records"))
    monkeypatch.setenv("MONGO_URI", "mongodb://mock")
    monkeypatch.setattr("pymongo.MongoClient", lambda *_a, **_k: client)
    run_dir = tmp_path / "run"
    result = ingestion.ingest(cfg, run_dir)
    assert result["source"] == "mongodb" and result["rows"] == 400


@mock_aws
def test_s3_sync_uploads_files(tmp_path):
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="demo")
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("y")
    assert storage.sync_to_s3(tmp_path, "netguard/run1", bucket="demo") == 2
    keys = [o["Key"] for o in boto3.client("s3", region_name="us-east-1")
            .list_objects_v2(Bucket="demo")["Contents"]]
    assert sorted(keys) == ["netguard/run1/a.txt", "netguard/run1/sub/b.txt"]


def test_s3_failure_does_not_lose_the_run(cfg, monkeypatch):
    def denied(*_a, **_k):
        raise PermissionError("AccessDenied simulated")

    monkeypatch.setattr(storage, "sync_to_s3", denied)
    out = pipeline.run(cfg)
    assert out["promoted"] is True
    assert "AccessDenied" in out["s3_error"] and out["s3_files_uploaded"] == 0
    assert (Path(cfg["paths"]["final_model_dir"]) / "model.joblib").exists()


@mock_aws
def test_sync_script_uploads_latest_run_and_model(cfg, monkeypatch):
    import runpy

    pipeline.run(cfg)
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="demo")
    monkeypatch.setenv("S3_BUCKET", "demo")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("NETGUARD_CONFIG", str(ROOT / "config" / "config.yaml"))
    monkeypatch.setattr("netguard.config.load_config", lambda *a, **k: cfg)
    runpy.run_path(str(ROOT / "scripts" / "sync_to_s3.py"), run_name="__main__")
    keys = [o["Key"] for o in boto3.client("s3", region_name="us-east-1")
            .list_objects_v2(Bucket="demo")["Contents"]]
    assert any(k.startswith("netguard/final_model/model.joblib") for k in keys)
    assert any("/ingested/train.csv" in k for k in keys)


def test_api_end_to_end(cfg, monkeypatch):
    pipeline.run(cfg)
    monkeypatch.setenv("NETGUARD_CONFIG", str(ROOT / "config" / "config.yaml"))
    import app as app_module

    app_module.cfg = cfg
    with TestClient(app_module.app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        rows = make(50, 9).drop(columns=["label"]).fillna(0)
        r = client.post("/predict", json={"records": rows.head(5).to_dict("records")})
        assert r.status_code == 200 and len(r.json()["predictions"]) == 5
        csv_bytes = rows.to_csv(index=False).encode()
        r = client.post("/predict/csv", files={"file": ("x.csv", csv_bytes, "text/csv")})
        assert r.json()["rows"] == 50
        bad = client.post("/predict", json={"records": [{"duration": 1}]})
        assert bad.status_code == 422
        assert "model_name" in client.get("/model-info").json()
