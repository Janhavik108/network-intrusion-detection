"""Stage 4: tune five classifiers, track them in MLflow, keep the CV winner."""
import json
import os
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import get_logger, resolve
from .transformation import build_preprocessor

log = get_logger("netguard.trainer")


def model_zoo(seed: int) -> dict:
    return {
        "random_forest": (RandomForestClassifier(random_state=seed, n_jobs=-1),
                          {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 12]}),
        "decision_tree": (DecisionTreeClassifier(random_state=seed),
                          {"clf__max_depth": [6, 12, None], "clf__min_samples_leaf": [1, 5]}),
        "gradient_boosting": (GradientBoostingClassifier(random_state=seed),
                              {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]}),
        "logistic_regression": (LogisticRegression(max_iter=1000),
                                {"clf__C": [0.1, 1.0, 10.0]}),
        "adaboost": (AdaBoostClassifier(random_state=seed),
                     {"clf__n_estimators": [50, 150], "clf__learning_rate": [0.5, 1.0]}),
    }


def _scores(pipe, X, y) -> dict:
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "false_positive_rate": fp / max(fp + tn, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _setup_mlflow(cfg: dict, run_dir: Path):
    uri = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{resolve(cfg, 'mlflow.db')}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    return uri


def train(cfg: dict, train_path: str, test_path: str, run_dir: Path) -> dict:
    seed = cfg["data"]["seed"]
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    tr, te = pd.read_csv(train_path), pd.read_csv(test_path)
    Xtr, ytr, Xte, yte = tr[feats], tr["target"], te[feats], te["target"]

    tracking_uri = _setup_mlflow(cfg, run_dir)
    metric = cfg["training"]["selection_metric"]
    cv = StratifiedKFold(cfg["training"]["cv_folds"], shuffle=True, random_state=seed)
    results, fitted = {}, {}

    with mlflow.start_run(run_name=run_dir.name) as parent:
        mlflow.log_params({"train_rows": len(tr), "test_rows": len(te), "selection_metric": metric})
        for name, (clf, grid) in model_zoo(seed).items():
            t0 = time.time()
            pipe = Pipeline([("prep", build_preprocessor(cfg)), ("clf", clone(clf))])
            search = GridSearchCV(pipe, grid, scoring=metric, cv=cv, n_jobs=-1).fit(Xtr, ytr)
            held_out = _scores(search.best_estimator_, Xte, yte)
            results[name] = {"cv_score": float(search.best_score_), "best_params": search.best_params_,
                             "test": held_out, "fit_seconds": round(time.time() - t0, 1)}
            fitted[name] = search.best_estimator_
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
                mlflow.log_metric(f"cv_{metric}", search.best_score_)
                mlflow.log_metrics({f"test_{k}": float(v) for k, v in held_out.items()})
            log.info("%-20s cv_%s=%.4f test_f1=%.4f test_recall=%.4f", name, metric,
                     search.best_score_, held_out["f1"], held_out["recall"])

        best = max(results, key=lambda n: results[n]["cv_score"])
        out = run_dir / "model"
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted[best], out / "model.joblib")
        meta = {
            "model_name": best,
            "features": {"numeric": cfg["features"]["numeric"], "categorical": cfg["features"]["categorical"]},
            "cv_score": results[best]["cv_score"],
            "selection_metric": metric,
            "test_metrics": results[best]["test"],
            "all_models": results,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": run_dir.name,
            "mlflow_run_id": parent.info.run_id,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        mlflow.set_tag("best_model", best)
        mlflow.log_artifact(str(out / "metadata.json"))
        mlflow.log_artifact(str(out / "model.joblib"))

    log.info("Best by CV %s: %s (held-out f1=%.4f)", metric, best, results[best]["test"]["f1"])
    return {"best_model": best, "model_dir": str(out), "metadata": meta, "tracking_uri": tracking_uri}
