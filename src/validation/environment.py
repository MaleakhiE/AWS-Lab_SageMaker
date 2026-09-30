"""Validate the local SageMaker execution environment and AWS identity."""

from __future__ import annotations

import sys


def validate_environment() -> dict[str, str]:
    """Return versions and caller identity, failing clearly when unavailable."""
    try:
        import boto3
        import sagemaker
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements first: python -m pip install -r requirements.txt"
        ) from exc

    session = boto3.Session()
    if not session.region_name:
        raise RuntimeError("AWS region is not configured; set AWS_REGION or configure AWS CLI.")

    identity = session.client("sts").get_caller_identity()
    return {
        "boto3_version": boto3.__version__,
        "sagemaker_version": sagemaker.__version__,
        "region": session.region_name,
        "account": identity["Account"],
        "caller_arn": identity["Arn"],
    }


def main() -> int:
    try:
        for key, value in validate_environment().items():
            print(f"{key}: {value}")
    except Exception as exc:  # noqa: BLE001 - CLI should report a concise failure.
        print(f"Environment validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
