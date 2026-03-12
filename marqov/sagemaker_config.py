"""
SageMaker configuration for quimb GPU benchmarking.

Reuses marqov-agent's AWS infrastructure (S3, IAM role, GPU instances).
"""

# Shared with marqov-agent
S3_BUCKET = "quantum8b-training-2026"
ROLE_ARN = "arn:aws:iam::090208085542:role/SageMakerExecutionRole-Quantum8B"
TRAINING_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker"
AWS_REGION = "us-east-1"

# quimb-specific
S3_PREFIX = "quimb-bench"
INSTANCE_TYPE = "ml.g5.12xlarge"  # 4x A10G, 96GB VRAM
VOLUME_SIZE_GB = 100
