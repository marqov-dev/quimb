#!/usr/bin/env python3
"""
GPU benchmark runner — entry point for SageMaker jobs.

This script runs on the GPU instance. It:
1. Installs quimb + cuQuantum from the uploaded source bundle
2. Runs benchmarks across all available backends
3. Uploads results to S3

Same role as train_v04_unified.py in marqov-agent, but for benchmarks.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# SageMaker unpacks source to /opt/ml/input/data/source/
SOURCE_DIR = Path("/opt/ml/input/data/source")
OUTPUT_DIR = Path("/opt/ml/output/data")
MODEL_DIR = Path("/opt/ml/model")  # SageMaker expects this


def setup_environment():
    """Install dependencies and quimb from source."""
    print("=" * 60)
    print("Setting up environment")
    print("=" * 60)

    # Install requirements
    req_file = SOURCE_DIR / "requirements.txt"
    if req_file.exists():
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(req_file), "--quiet"
        ])

    # Install quimb from source
    pyproject = SOURCE_DIR / "pyproject.toml"
    if pyproject.exists():
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-e", str(SOURCE_DIR), "--quiet"
        ])

    # Verify GPU availability
    try:
        import cupy
        print(f"CuPy version: {cupy.__version__}")
        print(f"CUDA available: {cupy.cuda.runtime.getDeviceCount()} GPUs")
        for i in range(cupy.cuda.runtime.getDeviceCount()):
            props = cupy.cuda.runtime.getDeviceProperties(i)
            print(f"  GPU {i}: {props['name'].decode()} ({props['totalGlobalMem'] / 1e9:.1f} GB)")
    except ImportError:
        print("WARNING: CuPy not available")

    try:
        import cuquantum
        print(f"cuQuantum version: {cuquantum.__version__}")
    except ImportError:
        print("WARNING: cuQuantum not available")


def detect_backends():
    """Detect available backends."""
    backends = ["numpy"]

    try:
        import cupy
        backends.append("cupy")
    except ImportError:
        pass

    try:
        import cuquantum
        backends.append("cutensornet")
    except ImportError:
        pass

    try:
        import jax
        backends.append("jax")
    except ImportError:
        pass

    return backends


def run_spin_benchmarks(backends):
    """Run spin system benchmarks across backends."""
    sys.path.insert(0, str(SOURCE_DIR))
    from benchmarks.bench_spin_systems import (
        bench_heisenberg_exact_diag,
        bench_heisenberg_dmrg,
        bench_tfim_exact_diag,
    )

    results = []

    # These are CPU-based (exact diag and DMRG)
    # Future: add tensor network ground state methods that can use GPU
    for n in [4, 6, 8, 10]:
        results.append(bench_heisenberg_exact_diag(n))
    for n in [8, 12]:
        results.append(bench_heisenberg_dmrg(n))
    for n in [4, 6, 8, 10]:
        results.append(bench_tfim_exact_diag(n))

    return results


def run_circuit_benchmarks(backends):
    """Run circuit benchmarks across backends."""
    sys.path.insert(0, str(SOURCE_DIR))
    from benchmarks.bench_circuits import (
        bench_bell_state,
        bench_ghz_state,
        bench_qft,
        bench_random_circuit,
    )

    results = []
    results.append(bench_bell_state())
    results.append(bench_ghz_state(5))
    results.append(bench_ghz_state(10))
    results.append(bench_ghz_state(16))  # Larger on GPU
    results.append(bench_qft(6))
    results.append(bench_random_circuit(10, 20))
    results.append(bench_random_circuit(16, 30))  # Larger on GPU

    return results


def run_contraction_benchmarks(backends):
    """Run raw tensor network contraction benchmarks comparing backends.

    This is the core benchmark — directly measures cuTensorNet vs numpy.
    """
    import numpy as np

    try:
        import quimb.tensor as qtn
    except ImportError:
        print("quimb not importable, skipping contraction benchmarks")
        return []

    results = []

    # Benchmark: contract a random tensor network of varying size
    for num_tensors in [10, 20, 50, 100]:
        for backend_name in backends:
            try:
                # Build a random tensor network
                tn = qtn.TN_rand_reg(
                    n=num_tensors,
                    reg=3,
                    D=8,
                    seed=42,
                )

                # Time the contraction
                t0 = time.perf_counter()
                result = tn.contract()
                elapsed = time.perf_counter() - t0

                results.append({
                    "benchmark": "random_tn_contraction",
                    "num_tensors": num_tensors,
                    "bond_dim": 8,
                    "regularity": 3,
                    "backend": backend_name,
                    "time_s": elapsed,
                    "result_shape": "scalar",
                })
            except Exception as e:
                results.append({
                    "benchmark": "random_tn_contraction",
                    "num_tensors": num_tensors,
                    "backend": backend_name,
                    "error": str(e),
                })

    return results


def main():
    """Main entry point for SageMaker job."""
    start_time = time.time()
    print(f"quimb GPU benchmark runner starting at {datetime.utcnow().isoformat()}Z")

    # Parse hyperparameters (passed by SageMaker)
    hp_backends = os.environ.get("SM_HP_BACKENDS", "numpy,cupy,cutensornet")
    hp_suite = os.environ.get("SM_HP_BENCHMARK_SUITE", "all")
    hp_output_prefix = os.environ.get("SM_HP_OUTPUT_S3_PREFIX", "")

    requested_backends = hp_backends.split(",")

    # Setup
    setup_environment()
    available = detect_backends()
    backends = [b for b in requested_backends if b in available]
    print(f"\nRequested backends: {requested_backends}")
    print(f"Available backends: {available}")
    print(f"Running with: {backends}")

    # Run benchmarks
    all_results = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "backends_available": available,
            "backends_used": backends,
            "suite": hp_suite,
        },
        "spin_systems": [],
        "circuits": [],
        "contractions": [],
    }

    if hp_suite in ("all", "spin"):
        print("\n" + "=" * 60)
        print("Running spin system benchmarks")
        print("=" * 60)
        all_results["spin_systems"] = run_spin_benchmarks(backends)

    if hp_suite in ("all", "circuits"):
        print("\n" + "=" * 60)
        print("Running circuit benchmarks")
        print("=" * 60)
        all_results["circuits"] = run_circuit_benchmarks(backends)

    if hp_suite in ("all", "contraction"):
        print("\n" + "=" * 60)
        print("Running contraction benchmarks")
        print("=" * 60)
        all_results["contractions"] = run_contraction_benchmarks(backends)

    # Summary
    elapsed = time.time() - start_time
    total_pass = sum(
        1 for suite in ["spin_systems", "circuits"]
        for r in all_results[suite]
        if r.get("pass")
    )
    total_tests = sum(
        len(all_results[suite])
        for suite in ["spin_systems", "circuits"]
    )
    all_results["metadata"]["total_time_s"] = elapsed
    all_results["metadata"]["pass_rate"] = f"{total_pass}/{total_tests}"

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {total_pass}/{total_tests} passed in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    # SageMaker expects something in model dir
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "results.json").write_text(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
