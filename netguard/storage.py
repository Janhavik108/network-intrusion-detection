"""Sync run artifacts to S3. A no-op when S3_BUCKET is not configured."""
import os
from pathlib import Path

from .config import get_logger

log = get_logger("netguard.storage")


def sync_to_s3(local_dir: Path, prefix: str, bucket: str | None = None) -> int:
    bucket = bucket or os.getenv("S3_BUCKET")
    if not bucket:
        log.info("S3_BUCKET not set, skipping artifact sync")
        return 0
    import boto3

    s3 = boto3.client("s3")
    count = 0
    for path in Path(local_dir).rglob("*"):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(local_dir).as_posix()}"
            s3.upload_file(str(path), bucket, key)
            count += 1
    log.info("Uploaded %d files to s3://%s/%s", count, bucket, prefix)
    return count
