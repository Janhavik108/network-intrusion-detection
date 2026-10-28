"""Run the full training pipeline: python main.py [--csv path/to/data.csv]"""
import argparse
import json

from netguard import pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Override the data source with a CSV (skips MongoDB)")
    args = ap.parse_args()
    print(json.dumps(pipeline.run(csv_path=args.csv), indent=2, default=str))
