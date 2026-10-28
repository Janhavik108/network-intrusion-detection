"""Stage 3: feature preprocessing.

The preprocessor is fitted on the training split only and saved on its own for
inspection. The trainer clones the unfitted spec inside each CV pipeline so
cross-validation folds never see statistics from their own validation data.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import get_logger

log = get_logger("netguard.transformation")


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, cfg["features"]["numeric"]),
        ("cat", categorical, cfg["features"]["categorical"]),
    ])


def transform(cfg: dict, train_path: str, run_dir: Path) -> dict:
    train = pd.read_csv(train_path)
    feature_cols = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    prep = build_preprocessor(cfg).fit(train[feature_cols])
    out = run_dir / "transformation"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(prep, out / "preprocessor.joblib")
    n_out = prep.transform(train[feature_cols].head(5)).shape[1]
    log.info("Preprocessor fitted on %d rows, %d input cols -> %d model features",
             len(train), len(feature_cols), n_out)
    return {"preprocessor_path": str(out / "preprocessor.joblib"), "n_model_features": int(n_out)}
