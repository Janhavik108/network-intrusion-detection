"""FastAPI service: batch scoring of network flow records plus on-demand retraining."""
import io
import threading
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from netguard import pipeline
from netguard.config import NetGuardError, get_logger, load_config
from netguard.predictor import Predictor

cfg = load_config()
log = get_logger("netguard.api", cfg)
state: dict[str, Any] = {"predictor": None, "train": {"status": "idle", "result": None}}
train_lock = threading.Lock()


def _load_model():
    try:
        state["predictor"] = Predictor(cfg)
        log.info("Model loaded: %s", state["predictor"].meta["model_name"])
    except NetGuardError as e:
        state["predictor"] = None
        log.warning(str(e))


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(title="NetGuard Intrusion Detection API", version="0.1.0", lifespan=lifespan)


class Records(BaseModel):
    records: list[dict[str, Any]]


def _require_model() -> Predictor:
    if state["predictor"] is None:
        raise HTTPException(503, "No model loaded. POST /train first.")
    return state["predictor"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["predictor"] is not None}


@app.get("/model-info")
def model_info():
    meta = _require_model().meta
    return {k: meta[k] for k in ("model_name", "trained_at", "selection_metric", "cv_score",
                                 "test_metrics", "features", "run_id")}


@app.post("/predict")
def predict(body: Records):
    if not body.records:
        raise HTTPException(422, "records must not be empty")
    try:
        return {"predictions": _require_model().predict(pd.DataFrame(body.records))}
    except NetGuardError as e:
        raise HTTPException(422, str(e))


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        preds = _require_model().predict(df)
    except NetGuardError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")
    n_attack = sum(p["label"] == "attack" for p in preds)
    return {"rows": len(preds), "attacks_flagged": n_attack, "predictions": preds}


def _train_job():
    try:
        state["train"] = {"status": "running", "result": None}
        state["train"] = {"status": "done", "result": pipeline.run(cfg)}
        _load_model()
    except Exception as e:
        log.exception("Training failed")
        state["train"] = {"status": "failed", "result": str(e)}
    finally:
        train_lock.release()


@app.post("/train", status_code=202)
def train(background: BackgroundTasks):
    if not train_lock.acquire(blocking=False):
        raise HTTPException(409, "Training already running")
    background.add_task(_train_job)
    return {"status": "started"}


@app.get("/train/status")
def train_status():
    return state["train"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
