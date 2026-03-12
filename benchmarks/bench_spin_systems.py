"""Benchmark: Spin system ground states with known analytical results.

These benchmarks verify correctness and measure performance across backends.
Each test computes a ground state energy and compares to a known reference value.
"""

import time

import numpy as np

import quimb as qu
import quimb.tensor as qtn


# =============================================================================
# Reference values (analytically known or numerically verified)
# =============================================================================

# Heisenberg chain ground state energies (from Bethe ansatz / exact diag)
HEISENBERG_REFERENCE = {
    4: -1.6160254038,
    6: -2.4935771339,
    8: -3.3749325987,
    10: -4.2580352073,
    12: -5.1420906328,
}

# Transverse field Ising model at critical point (J=1, h=1)
TFIM_CRITICAL_REFERENCE = {
    4: -2.0941996592,
    6: -3.1577410691,
    8: -4.2212853733,
    10: -5.2848297789,
}


def bench_heisenberg_exact_diag(n_sites=8):
    """Ground state of Heisenberg chain via exact diagonalization."""
    t0 = time.perf_counter()
    H = qu.ham_heis(n_sites, sparse=True)
    energy = qu.groundenergy(H)
    elapsed = time.perf_counter() - t0

    ref = HEISENBERG_REFERENCE[n_sites]
    error = abs(energy - ref)

    return {
        "system": "heisenberg",
        "method": "exact_diag",
        "n_sites": n_sites,
        "energy": energy,
        "reference": ref,
        "error": error,
        "pass": error < 1e-6,
        "time_s": elapsed,
    }


def bench_heisenberg_dmrg(n_sites=12, bond_dim=32):
    """Ground state of Heisenberg chain via DMRG."""
    t0 = time.perf_counter()
    builder = qtn.SpinHam1D(S=0.5)
    builder += 1.0, "X", "X"
    builder += 1.0, "Y", "Y"
    builder += 1.0, "Z", "Z"
    H = builder.build_mpo(n_sites)

    dmrg = qtn.DMRG2(H, bond_dims=[bond_dim])
    dmrg.solve(max_sweeps=20, verbosity=0)
    energy = dmrg.energy
    elapsed = time.perf_counter() - t0

    ref = HEISENBERG_REFERENCE.get(n_sites)
    error = abs(energy - ref) if ref else None

    return {
        "system": "heisenberg",
        "method": "dmrg",
        "n_sites": n_sites,
        "bond_dim": bond_dim,
        "energy": energy,
        "reference": ref,
        "error": error,
        "pass": error < 1e-4 if error is not None else None,
        "time_s": elapsed,
    }


def bench_tfim_exact_diag(n_sites=8, h=1.0):
    """Ground state of transverse field Ising model at critical point."""
    t0 = time.perf_counter()
    H = qu.ham_ising(n_sites, jz=1.0, bx=h, sparse=True)
    energy = qu.groundenergy(H)
    elapsed = time.perf_counter() - t0

    ref = TFIM_CRITICAL_REFERENCE.get(n_sites)
    error = abs(energy - ref) if ref else None

    return {
        "system": "tfim_critical",
        "method": "exact_diag",
        "n_sites": n_sites,
        "h_field": h,
        "energy": energy,
        "reference": ref,
        "error": error,
        "pass": error < 1e-6 if error is not None else None,
        "time_s": elapsed,
    }


def run_all():
    """Run all spin system benchmarks and print results."""
    results = []

    # Exact diag benchmarks
    for n in [4, 6, 8, 10]:
        results.append(bench_heisenberg_exact_diag(n))

    # DMRG benchmarks
    for n in [8, 12]:
        results.append(bench_heisenberg_dmrg(n))

    # TFIM benchmarks
    for n in [4, 6, 8, 10]:
        results.append(bench_tfim_exact_diag(n))

    # Print summary
    print(f"\n{'System':<20} {'Method':<12} {'N':<5} {'Energy':<14} {'Ref':<14} {'Error':<12} {'Pass':<6} {'Time':<8}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['system']:<20} {r['method']:<12} {r['n_sites']:<5} "
            f"{r['energy']:<14.8f} {r.get('reference', 'N/A')!s:<14} "
            f"{r.get('error', 'N/A')!s:<12} {r.get('pass', 'N/A')!s:<6} "
            f"{r['time_s']:<8.3f}"
        )

    passed = sum(1 for r in results if r.get("pass"))
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    return results


if __name__ == "__main__":
    run_all()
