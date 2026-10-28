"""Load a CSV into MongoDB so the pipeline can ingest from it.

Uploads in batches with retries. Each row gets a deterministic _id, so a retried
batch cannot create duplicates.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, BulkWriteError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from netguard.config import load_config, resolve  # noqa: E402

BATCH = 5000
RETRIES = 4


def insert_batch(coll, chunk):
    for attempt in range(1, RETRIES + 1):
        try:
            coll.insert_many(chunk, ordered=False)
            return
        except BulkWriteError as e:
            # rows already written by an earlier attempt show up as duplicate keys; that is fine
            if all(err.get("code") == 11000 for err in e.details["writeErrors"]):
                return
            raise
        except AutoReconnect:
            if attempt == RETRIES:
                raise
            time.sleep(2 * attempt)


if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=cfg["data"]["csv_path"])
    ap.add_argument("--replace", action="store_true", help="Drop the collection first")
    a = ap.parse_args()

    uri = os.environ.get("MONGO_URI")
    if not uri:
        sys.exit("Set MONGO_URI first (see .env.example)")
    df = pd.read_csv(resolve(cfg, a.csv))
    records = df.to_dict("records")
    for i, r in enumerate(records):
        r["_id"] = i

    coll = MongoClient(uri)[cfg["mongo"]["database"]][cfg["mongo"]["collection"]]
    if a.replace:
        coll.drop()
    for start in range(0, len(records), BATCH):
        insert_batch(coll, records[start:start + BATCH])
        print(f"  uploaded {min(start + BATCH, len(records))}/{len(records)}", flush=True)

    stored = coll.count_documents({})
    name = f"{cfg['mongo']['database']}.{cfg['mongo']['collection']}"
    if stored != len(records):
        sys.exit(f"Expected {len(records)} documents in {name} but found {stored}. "
                 "Re-run with --replace.")
    print(f"Inserted {len(records)} records into {name}")