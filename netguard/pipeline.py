"""Orchestrates the four stages, promotes the model if it beats the champion, syncs to S3."""
import json
import shutil
import time

from . import ingestion, storage, trainer, transformation, validation
from .config import get_logger, load_config, resolve

log = get_logger("netguard.pipeline")


def _promote(cfg: dict, model_dir, metadata: dict) -> dict:
    final = resolve(cfg, cfg["paths"]["final_model_dir"])
    current = final / "metadata.json"
    metric = metadata["selection_metric"]
    delta = cfg["training"]["promotion_min_delta"]
    if current.exists():
        old = json.loads(current.read_text())
        if metadata["test_metrics"]["f1"] < old["test_metrics"]["f1"] + delta:
            log.info("Candidate f1=%.4f did not beat champion f1=%.4f, keeping champion",
                     metadata["test_metrics"]["f1"], old["test_metrics"]["f1"])
            return {"promoted": False, "champion_f1": old["test_metrics"]["f1"]}
    final.mkdir(parents=True, exist_ok=True)
    for name in ("model.joblib", "metadata.json"):
        shutil.copy2(model_dir / name, final / name)
    log.info("Promoted %s to %s (selection metric: %s)", metadata["model_name"], final, metric)
    return {"promoted": True}


def run(cfg: dict | None = None, csv_path: str | None = None) -> dict:
    from pathlib import Path

    cfg = cfg or load_config()
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = resolve(cfg, cfg["paths"]["artifacts_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ing = ingestion.ingest(cfg, run_dir, csv_path)
    val = validation.validate(cfg, ing["train_path"], ing["test_path"], run_dir)
    tfm = transformation.transform(cfg, ing["train_path"], run_dir)
    trn = trainer.train(cfg, ing["train_path"], ing["test_path"], run_dir)
    promo = _promote(cfg, Path(trn["model_dir"]), trn["metadata"])
    uploaded, s3_error = 0, None
    try:
        uploaded = storage.sync_to_s3(run_dir, f"{cfg['s3']['prefix']}/{run_id}")
        if promo["promoted"]:
            storage.sync_to_s3(resolve(cfg, cfg["paths"]["final_model_dir"]),
                               f"{cfg['s3']['prefix']}/final_model")
    except Exception as e:  # training results are already saved locally, so do not lose the run
        s3_error = str(e)
        log.error("S3 sync failed, results are kept locally. Fix access, then run "
                  "scripts/sync_to_s3.py. Error: %s", e)

    summary = {
        "run_id": run_id, "data_source": ing["source"], "rows": ing["rows"],
        "validation_warnings": val["warnings"], "best_model": trn["best_model"],
        "test_metrics": trn["metadata"]["test_metrics"], "promoted": promo["promoted"],
        "s3_files_uploaded": uploaded, "s3_error": s3_error,
        "mlflow_tracking_uri": trn["tracking_uri"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
