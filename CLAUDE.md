# Marqov × quimb: Heterogeneous Tensor Network Acceleration

## Project Goal

Use AI agents to extend quimb with intelligent hardware routing for tensor network contractions — GPU acceleration via NVIDIA cuQuantum/cuTensorNet, hybrid CPU/GPU partitioning, and automatic backend selection.

This is a tracked experiment. Every commit counts. Every test matters. We are measuring:
- Total commits
- Lines added
- Test count growth
- Benchmark pass rate over time
- Wall clock days

## Architecture

quimb uses `cotengra` for contraction path optimization and `autoray` for backend dispatch (numpy, jax, torch, cupy). Our work adds:

1. **cuTensorNet backend** — plug NVIDIA's optimized contraction engine into quimb's pipeline
2. **Contraction router** — analyze tensor network structure and pick optimal backend automatically
3. **Hybrid partitioning** — split networks so subnetworks run on optimal hardware
4. **Benchmark suite** — known physics results (spin chains, Ising model, molecular H2) as regression tests
5. **Marqov Router integration** — the Posterior → ExecutionPlan pipeline for tensor networks

## Commit Categories

Every commit message MUST be prefixed with a category:

- `backend:` — cuTensorNet/cuQuantum integration
- `router:` — intelligent backend selection and routing logic
- `hybrid:` — hybrid CPU/GPU partitioning
- `bench:` — benchmark suite and regression tests
- `test:` — unit tests and test infrastructure
- `perf:` — performance optimization
- `fix:` — bug fixes
- `docs:` — documentation and engineering log
- `infra:` — CI, tooling, project scaffolding

Example: `backend: add cuTensorNet contraction path for einsum operations`

## Key Files

- `quimb/tensor/contraction.py` — contraction strategy dispatch (main integration point)
- `quimb/tensor/tensor_core.py` — core Tensor class
- `quimb/tensor/circuit.py` — quantum circuit simulation
- `quimb/tensor/optimize.py` — tensor network optimization
- `quimb/tensor/interface.py` — backend interface via autoray

## Development Rules

1. Always run existing tests before and after changes: `python -m pytest tests/ -x -q`
2. Log every agent iteration in `engineering-log/`
3. One logical change per commit — small, atomic commits
4. Every new feature needs a test
5. Benchmark results go in `benchmarks/results/`
6. Do not break upstream compatibility — our changes are additive
