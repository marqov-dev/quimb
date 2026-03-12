# Benchmark Suite

Regression benchmarks with known analytical or numerically-verified results.
These serve as our "sv-tests" — a compliance score we climb over time.

## Categories

### Spin Systems
- Heisenberg chain (exact ground state energy for small N)
- Transverse field Ising model (analytical in 1D)
- XXZ model

### Molecular Systems
- H2 molecule (FCI energy: -1.1373 Ha at equilibrium)
- HeH+ (reference from PySCF)

### Quantum Circuits
- Bell state preparation (exact statevector)
- GHZ states (exact)
- Random circuit sampling (compare to exact statevector sim)
- QFT circuits

### Tensor Network Operations
- MPS compression accuracy
- DMRG convergence on known systems
- TEBD time evolution accuracy

## Metrics Tracked
- Correctness (within tolerance of reference)
- Wall time per backend (numpy, jax, cupy, cutensornet)
- Peak memory
- Contraction FLOPS
