"""Upload the newest run in Artifacts/ and final_model/ to S3 without retraining.

Use this after fixing S3 credentials or permissions when a run finished but the upload failed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from netguard import storage  # noqa: E402
from netguard.config import load_config, resolve  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    runs = sorted(resolve(cfg, cfg["paths"]["artifacts_dir"]).glob("run_*"))
    if not runs:
        sys.exit("No runs found in the artifacts folder. Run main.py first.")
    latest = runs[-1]
    prefix = cfg["s3"]["prefix"]
    n_run = storage.sync_to_s3(latest, f"{prefix}/{latest.name}")
    n_model = storage.sync_to_s3(resolve(cfg, cfg["paths"]["final_model_dir"]), f"{prefix}/final_model")
    if n_run == 0 and n_model == 0:
        sys.exit("Nothing uploaded. Is S3_BUCKET set in .env?")
    print(f"Uploaded {n_run} files from {latest.name} and {n_model} model files")
