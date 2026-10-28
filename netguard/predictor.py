import json

import joblib
import pandas as pd

from .config import NetGuardError, resolve


class Predictor:
    def __init__(self, cfg: dict):
        final = resolve(cfg, cfg["paths"]["final_model_dir"])
        if not (final / "model.joblib").exists():
            raise NetGuardError("No trained model found. Run `python main.py` or POST /train first.")
        self.model = joblib.load(final / "model.joblib")
        self.meta = json.loads((final / "metadata.json").read_text())
        feats = self.meta["features"]
        self.columns = feats["numeric"] + feats["categorical"]

    def predict(self, df: pd.DataFrame) -> list[dict]:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise NetGuardError(f"Missing required columns: {missing}")
        X = df[self.columns]
        proba = self.model.predict_proba(X)[:, 1]
        return [{"label": "attack" if p >= 0.5 else "benign", "attack_probability": round(float(p), 4)}
                for p in proba]
