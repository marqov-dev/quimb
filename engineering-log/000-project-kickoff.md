# Iteration 000: Project Kickoff

**Date:** 2026-03-12
**Agent:** Claude Opus 4.6 (manual session)
**Duration:** ~60 min

## Context

Forked jcmgray/quimb to marqov-dev/quimb. Goal: use AI agents to add heterogeneous hardware routing for tensor network contractions — GPU acceleration via cuQuantum, hybrid CPU/GPU partitioning, automatic backend selection.

Inspired by Normal Computing's CIRCT experiment (580K lines, 43 days, 2,968 commits).

## Baseline Metrics

- **Source files:** 109 Python files
- **Source lines:** 98,078
- **Test files:** 54
- **Test lines:** 14,779
- **Upstream commits:** 2,359

## Key Findings

- quimb uses `cotengra` for contraction path optimization
- Backend dispatch via `autoray` (numpy, jax, torch, cupy)
- Contraction strategy is configurable via `set_contract_strategy()`
- `quimb/tensor/contraction.py` is the main integration point
- `quimb/tensor/tensor_core.py` is the core Tensor class (~5K lines)
- Circuit simulation in `quimb/tensor/circuit.py`

## Integration Points for cuQuantum

1. `contraction.py` — strategy dispatch, can add cuTensorNet as a strategy
2. `autoray` backend — cupy arrays already partially supported
3. `cotengra` — already has some cuTensorNet support for path finding, but not for execution
4. `tensor_core.py` contract method — where contraction actually happens

## Plan

1. Set up scaffolding and CI (this iteration)
2. Add cuTensorNet as a contraction backend
3. Build benchmark suite with known physics results
4. Add routing logic (analyze network → pick backend)
5. Add hybrid partitioning
6. Integrate Marqov Router types

## Blockers

None yet. Need GPU access for cuQuantum testing.
