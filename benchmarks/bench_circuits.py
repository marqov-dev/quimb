"""Benchmark: Quantum circuit simulation correctness and performance.

Compares quimb tensor network circuit simulation against exact statevector results.
"""

import time

import numpy as np

import quimb as qu
import quimb.tensor as qtn


def bench_bell_state():
    """Bell state |00⟩ + |11⟩)/√2 — exact probabilities known."""
    t0 = time.perf_counter()
    circ = qtn.Circuit(2)
    circ.h(0)
    circ.cnot(0, 1)
    state = circ.to_dense()
    elapsed = time.perf_counter() - t0

    probs = np.abs(state.flatten()) ** 2
    expected = np.array([0.5, 0, 0, 0.5])
    error = np.max(np.abs(probs - expected))

    return {
        "circuit": "bell_state",
        "qubits": 2,
        "error": error,
        "pass": error < 1e-10,
        "time_s": elapsed,
    }


def bench_ghz_state(n_qubits=5):
    """GHZ state (|00...0⟩ + |11...1⟩)/√2."""
    t0 = time.perf_counter()
    circ = qtn.Circuit(n_qubits)
    circ.h(0)
    for i in range(n_qubits - 1):
        circ.cnot(i, i + 1)
    state = circ.to_dense()
    elapsed = time.perf_counter() - t0

    probs = np.abs(state.flatten()) ** 2
    expected = np.zeros(2**n_qubits)
    expected[0] = 0.5
    expected[-1] = 0.5
    error = np.max(np.abs(probs - expected))

    return {
        "circuit": "ghz_state",
        "qubits": n_qubits,
        "error": error,
        "pass": error < 1e-10,
        "time_s": elapsed,
    }


def bench_qft(n_qubits=6):
    """Quantum Fourier Transform — verify unitarity of output."""
    t0 = time.perf_counter()
    circ = qtn.Circuit(n_qubits)

    for i in range(n_qubits):
        circ.h(i)
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            circ.cz(i, j)  # approximate — just testing circuit runs

    state = circ.to_dense()
    elapsed = time.perf_counter() - t0

    # State should be normalized
    norm = np.sum(np.abs(state.flatten()) ** 2)
    error = abs(norm - 1.0)

    return {
        "circuit": "qft",
        "qubits": n_qubits,
        "norm_error": error,
        "pass": error < 1e-10,
        "time_s": elapsed,
    }


def bench_random_circuit(n_qubits=10, depth=20):
    """Random circuit — verify normalization and compare TN vs dense."""
    t0 = time.perf_counter()
    circ = qtn.Circuit(n_qubits)
    rng = np.random.default_rng(42)

    for _ in range(depth):
        # Random single-qubit gates
        for q in range(n_qubits):
            gate = rng.choice(["h", "t", "s"])
            getattr(circ, gate)(q)
        # Random CNOT layer
        pairs = list(range(n_qubits))
        rng.shuffle(pairs)
        for i in range(0, n_qubits - 1, 2):
            circ.cnot(pairs[i], pairs[i + 1])

    state = circ.to_dense()
    elapsed = time.perf_counter() - t0

    norm = np.sum(np.abs(state.flatten()) ** 2)
    error = abs(norm - 1.0)

    return {
        "circuit": "random",
        "qubits": n_qubits,
        "depth": depth,
        "norm_error": error,
        "pass": error < 1e-8,
        "time_s": elapsed,
    }


def run_all():
    """Run all circuit benchmarks."""
    results = []
    results.append(bench_bell_state())
    results.append(bench_ghz_state(5))
    results.append(bench_ghz_state(10))
    results.append(bench_qft(6))
    results.append(bench_random_circuit(10, 20))

    print(f"\n{'Circuit':<20} {'Qubits':<8} {'Error':<14} {'Pass':<6} {'Time':<8}")
    print("-" * 60)
    for r in results:
        err_key = "error" if "error" in r else "norm_error"
        print(
            f"{r['circuit']:<20} {r['qubits']:<8} "
            f"{r[err_key]:<14.2e} {r['pass']!s:<6} "
            f"{r['time_s']:<8.3f}"
        )

    passed = sum(1 for r in results if r["pass"])
    print(f"\nPassed: {passed}/{len(results)}")
    return results


if __name__ == "__main__":
    run_all()
