"""Upload processed data to AWS S3.

Handles parquet file uploads with optional partitioning by state/year.
"""

from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger

from src.config import AWS_ACCESS_KEY_ID, AWS_REGION, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME


def get_s3_client():
    """Create an S3 client using configured credentials."""
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        return boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
    # Fall back to default credential chain (env vars, IAM role, etc.)
    return boto3.client("s3", region_name=AWS_REGION)


def upload_file(local_path: Path, s3_key: str, bucket: str = S3_BUCKET_NAME) -> bool:
    """Upload a single file to S3.

    Args:
        local_path: Path to the local file.
        s3_key: Destination key in the S3 bucket.
        bucket: S3 bucket name.

    Returns:
        True if upload succeeded, False otherwise.
    """
    try:
        client = get_s3_client()
        client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"Uploaded {local_path} → s3://{bucket}/{s3_key}")
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        return False
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        return False


def upload_data_dir(data_dir: Path, prefix: str = "raw", bucket: str = S3_BUCKET_NAME) -> int:
    """Upload all parquet files from a directory to S3.

    Args:
        data_dir: Local directory containing parquet files.
        prefix: S3 key prefix (e.g., 'raw' or 'processed').
        bucket: S3 bucket name.

    Returns:
        Number of files successfully uploaded.
    """
    uploaded = 0
    for path in data_dir.glob("*.parquet"):
        s3_key = f"{prefix}/{path.name}"
        if upload_file(path, s3_key, bucket):
            uploaded += 1
    logger.info(f"Uploaded {uploaded} files to s3://{bucket}/{prefix}/")
    return uploaded
