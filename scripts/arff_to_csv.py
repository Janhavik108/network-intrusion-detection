"""Convert NSL-KDD ARFF files to CSV and write a matching pipeline config.

    python scripts/arff_to_csv.py --train KDDTrain+.arff --test KDDTest+.arff

Outputs data/raw/nslkdd_train.csv, data/raw/nslkdd_test.csv and
config/config.nslkdd.yaml. Run the pipeline with:

    NETGUARD_CONFIG=config/config.nslkdd.yaml python main.py
"""
import argparse
import csv
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def read_arff(path) -> tuple[pd.DataFrame, dict]:
    names, types, rows, in_data = [], [], [], False
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if not in_data:
                if low.startswith("@attribute"):
                    rest = line[len("@attribute"):].strip()
                    if rest[0] in "'\"":
                        end = rest.index(rest[0], 1)
                        name, typ = rest[1:end], rest[end + 1:].strip()
                    else:
                        name, _, typ = rest.partition(" ")
                        typ = typ.strip()
                    names.append(name)
                    types.append(typ)
                elif low.startswith("@data"):
                    in_data = True
            else:
                rows.append(next(csv.reader([line], quotechar="'", skipinitialspace=True)))
    good = [r for r in rows if len(r) == len(names)]
    if len(good) != len(rows):
        print(f"warning: skipped {len(rows) - len(good)} malformed rows in {path}")
    df = pd.DataFrame(good, columns=names)
    return df, dict(zip(names, types))


def convert(train_arff, test_arff=None, out_dir="data/raw", config_out="config/config.nslkdd.yaml",
            target="class", base_config="config/config.yaml") -> Path:
    out_dir, config_out = Path(out_dir), Path(config_out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if not config_out.is_absolute():
        config_out = ROOT / config_out
    out_dir.mkdir(parents=True, exist_ok=True)

    train, types = read_arff(train_arff)
    numeric, categorical = [], []
    for col, typ in types.items():
        if col == target:
            continue
        if typ.startswith("{"):
            values = {v.strip().strip("'\"") for v in typ.strip("{}").split(",")}
            (numeric if values <= {"0", "1"} else categorical).append(col)
        else:
            numeric.append(col)

    def clean(df):
        df = df.copy()
        for c in numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in categorical + [target]:
            df[c] = df[c].astype(str).str.strip().str.strip("'\"")
        return df

    train = clean(train)
    train_csv = out_dir / "nslkdd_train.csv"
    train.to_csv(train_csv, index=False)
    test_csv = None
    if test_arff:
        test, _ = read_arff(test_arff)
        test_csv = out_dir / "nslkdd_test.csv"
        clean(test).to_csv(test_csv, index=False)

    base = Path(base_config) if Path(base_config).is_absolute() else ROOT / base_config
    cfg = yaml.safe_load(base.read_text())
    cfg["data"].update({"csv_path": str(train_csv.relative_to(ROOT)) if train_csv.is_relative_to(ROOT) else str(train_csv),
                        "target": target, "benign_labels": ["normal"]})
    if test_csv:
        cfg["data"]["test_csv_path"] = (str(test_csv.relative_to(ROOT)) if test_csv.is_relative_to(ROOT)
                                        else str(test_csv))
    cfg["features"] = {"numeric": numeric, "categorical": categorical}
    config_out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"train rows={len(train)} numeric={len(numeric)} categorical={categorical}")
    print(f"label counts: {train[target].value_counts().to_dict()}")
    print(f"wrote {train_csv}, {test_csv}, {config_out}")
    return config_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="e.g. KDDTrain+.arff")
    ap.add_argument("--test", help="e.g. KDDTest+.arff (used as the fixed test set)")
    ap.add_argument("--target", default="class")
    a = ap.parse_args()
    convert(a.train, a.test, target=a.target)
