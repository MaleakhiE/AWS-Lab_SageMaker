"""Validate the local SageMaker execution environment and AWS identity."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version


def _package_version(package_name: str, module: object) -> str:
    """Read a distribution version even when the module omits __version__."""
    module_version = getattr(module, "__version__", None)
    if module_version:
        return str(module_version)
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


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
        "boto3_version": _package_version("boto3", boto3),
        "sagemaker_version": _package_version("sagemaker", sagemaker),
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
