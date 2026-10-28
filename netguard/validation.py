"""Stage 2: schema, quality and drift checks on the train/test split."""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from .config import DataValidationError, get_logger

log = get_logger("netguard.validation")


def _drift(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict:
    p_thresh = cfg["validation"]["drift_p_value"]
    report = {}
    for col in cfg["features"]["numeric"]:
        p = float(ks_2samp(train[col].dropna(), test[col].dropna()).pvalue)
        report[col] = {"test": "ks", "p_value": p, "drifted": p < p_thresh}
    for col in cfg["features"]["categorical"]:
        table = pd.concat(
            [train[col].astype(str).value_counts(), test[col].astype(str).value_counts()], axis=1
        ).fillna(0)
        p = float(chi2_contingency(table.values)[1]) if table.shape[0] > 1 else 1.0
        report[col] = {"test": "chi2", "p_value": p, "drifted": p < p_thresh}
    return report


def validate(cfg: dict, train_path: str, test_path: str, run_dir: Path) -> dict:
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    numeric, categorical = cfg["features"]["numeric"], cfg["features"]["categorical"]
    expected = numeric + categorical + ["target"]
    errors, warnings = [], []

    for name, df in (("train", train), ("test", test)):
        missing_cols = [c for c in expected if c not in df.columns]
        if missing_cols:
            errors.append(f"{name}: missing columns {missing_cols}")
            continue
        for c in numeric:
            if not pd.api.types.is_numeric_dtype(df[c]):
                errors.append(f"{name}: column '{c}' is not numeric")
        frac = df[expected].isna().mean()
        for c, f in frac.items():
            if f > cfg["validation"]["max_missing_fraction"]:
                errors.append(f"{name}: column '{c}' is {f:.0%} missing")
        if df["target"].nunique() < 2:
            errors.append(f"{name}: only one class present")
        dup = float(df.duplicated().mean())
        if dup > 0.2:
            warnings.append(f"{name}: {dup:.0%} duplicate rows")

    drift = {}
    if not errors:
        drift = _drift(train, test, cfg)
        frac_drifted = sum(d["drifted"] for d in drift.values()) / len(drift)
        if frac_drifted > cfg["validation"]["max_drifted_fraction"]:
            msg = f"{frac_drifted:.0%} of features drifted between train and test"
            (errors if cfg["validation"]["fail_on_drift"] else warnings).append(msg)

    report = {"passed": not errors, "errors": errors, "warnings": warnings, "drift": drift}
    out = run_dir / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    log.info("Validation passed=%s errors=%d warnings=%d", report["passed"], len(errors), len(warnings))
    if errors:
        raise DataValidationError("; ".join(errors))
    return {"report_path": str(out / "report.json"), "warnings": warnings}
