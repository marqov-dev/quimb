#!/usr/bin/env python3
"""
SageMaker launcher for quimb GPU benchmarks.

Runs tensor network benchmarks across backends (numpy, cupy, cuTensorNet)
on GPU instances. Reuses marqov-agent's SageMaker infrastructure.

Subcommands:
  prepare    Upload quimb fork + benchmarks to S3
  bench      Launch GPU benchmark job
  status     Check job status
  results    Download and display results

Usage:
  python marqov/launch_benchmarks.py prepare
  python marqov/launch_benchmarks.py bench
  python marqov/launch_benchmarks.py status --job <job-name>
  python marqov/launch_benchmarks.py results --job <job-name>
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from sagemaker_config import (
    S3_BUCKET, S3_PREFIX, ROLE_ARN, TRAINING_IMAGE,
    AWS_REGION, INSTANCE_TYPE, VOLUME_SIZE_GB,
)


sagemaker = boto3.client("sagemaker", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)


def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def upload_source_bundle(s3_key_prefix: str) -> str:
    """Bundle quimb source + benchmarks into tar.gz and upload to S3.

    Same pattern as marqov-agent's upload_scripts().
    """
    repo_root = Path(__file__).parent.parent

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            # Include quimb source
            tar.add(repo_root / "quimb", arcname="quimb")
            # Include benchmarks
            tar.add(repo_root / "benchmarks", arcname="benchmarks")
            # Include marqov router
            tar.add(repo_root / "marqov", arcname="marqov")
            # Include the GPU runner script
            tar.add(repo_root / "marqov" / "gpu_runner.py", arcname="gpu_runner.py")
            # Include requirements
            tar.add(repo_root / "marqov" / "requirements-gpu.txt", arcname="requirements.txt")
            # Include setup
            tar.add(repo_root / "pyproject.toml", arcname="pyproject.toml")

        s3_key = f"{s3_key_prefix}/sourcedir.tar.gz"
        print(f"Uploading source bundle to s3://{S3_BUCKET}/{s3_key}")
        s3.upload_file(tmp_path, S3_BUCKET, s3_key)
        return f"s3://{S3_BUCKET}/{s3_key}"
    finally:
        os.unlink(tmp_path)


def cmd_prepare(args):
    """Upload source bundle to S3."""
    s3_uri = upload_source_bundle(f"{S3_PREFIX}/source")
    print(f"Source bundle uploaded: {s3_uri}")


def cmd_bench(args):
    """Launch GPU benchmark job on SageMaker."""
    ts = timestamp()
    job_name = f"quimb-bench-{ts}"

    # Upload fresh source
    source_uri = upload_source_bundle(f"{S3_PREFIX}/source")

    # Hyperparameters control which benchmarks run
    hyperparameters = {
        "backends": args.backends or "numpy,cupy,cutensornet",
        "benchmark_suite": args.suite or "all",
        "output_s3_prefix": f"{S3_PREFIX}/results/{ts}",
    }

    print(f"\nLaunching benchmark job: {job_name}")
    print(f"  Instance: {INSTANCE_TYPE}")
    print(f"  Backends: {hyperparameters['backends']}")
    print(f"  Suite: {hyperparameters['benchmark_suite']}")

    sagemaker.create_training_job(
        TrainingJobName=job_name,
        HyperParameters={k: str(v) for k, v in hyperparameters.items()},
        AlgorithmSpecification={
            "TrainingImage": TRAINING_IMAGE,
            "TrainingInputMode": "File",
        },
        RoleArn=ROLE_ARN,
        InputDataConfig=[
            {
                "ChannelName": "source",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": source_uri,
                    }
                },
            }
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{S3_BUCKET}/{S3_PREFIX}/output/",
        },
        ResourceConfig={
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": 1,
            "VolumeSizeInGB": VOLUME_SIZE_GB,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 7200,  # 2 hour max
        },
    )

    print(f"\nJob submitted: {job_name}")
    print(f"Check status: python marqov/launch_benchmarks.py status --job {job_name}")


def cmd_status(args):
    """Check job status."""
    try:
        response = sagemaker.describe_training_job(TrainingJobName=args.job)
        status = response["TrainingJobStatus"]
        print(f"Job: {args.job}")
        print(f"Status: {status}")

        if status == "Completed":
            print(f"Duration: {response.get('TrainingTimeInSeconds', '?')}s")
            print(f"Output: {response['OutputDataConfig']['S3OutputPath']}")
        elif status == "Failed":
            print(f"Failure: {response.get('FailureReason', 'Unknown')}")
        elif status == "InProgress":
            created = response.get("CreationTime")
            if created:
                elapsed = (datetime.now(created.tzinfo) - created).total_seconds()
                print(f"Running for: {elapsed:.0f}s")
    except ClientError as e:
        print(f"Error: {e}")


def cmd_results(args):
    """Download and display benchmark results."""
    # List result files in S3
    prefix = f"{S3_PREFIX}/results/"
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

    if "Contents" not in response:
        print("No results found.")
        return

    for obj in sorted(response["Contents"], key=lambda x: x["LastModified"], reverse=True):
        key = obj["Key"]
        if key.endswith(".json"):
            result = s3.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(result["Body"].read().decode())
            print(f"\n{'='*60}")
            print(f"Result: {key}")
            print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="quimb GPU benchmark launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    subparsers.add_parser("prepare", help="Upload source to S3")

    # bench
    bench_parser = subparsers.add_parser("bench", help="Launch GPU benchmark")
    bench_parser.add_argument("--backends", type=str, default=None,
                              help="Comma-separated backends (numpy,cupy,cutensornet)")
    bench_parser.add_argument("--suite", type=str, default=None,
                              help="Benchmark suite (spin,circuits,contraction,all)")

    # status
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("--job", required=True, help="Job name")

    # results
    results_parser = subparsers.add_parser("results", help="Download results")
    results_parser.add_argument("--job", required=False, help="Specific job name")

    args = parser.parse_args()

    commands = {
        "prepare": cmd_prepare,
        "bench": cmd_bench,
        "status": cmd_status,
        "results": cmd_results,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
