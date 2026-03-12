"""
Tensor network contraction router.

Analyzes tensor network structure and selects optimal contraction
backend and strategy. This is the core intelligence layer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Backend(Enum):
    """Available contraction backends."""
    NUMPY = "numpy"
    CUPY = "cupy"
    CUTENSORNET = "cutensornet"
    JAX = "jax"
    TORCH = "torch"


class RoutingReason(Enum):
    """Why a particular backend was selected."""
    NETWORK_TOO_SMALL = "network_too_small"  # CPU is faster for small networks
    NETWORK_LARGE = "network_large"  # GPU wins for large contractions
    MEMORY_CONSTRAINED = "memory_constrained"  # network won't fit in GPU memory
    BACKEND_UNAVAILABLE = "backend_unavailable"  # requested backend not installed
    USER_OVERRIDE = "user_override"  # user explicitly chose
    BENCHMARK_DATA = "benchmark_data"  # historical performance data


@dataclass
class NetworkProfile:
    """Structural profile of a tensor network for routing decisions."""
    num_tensors: int = 0
    num_indices: int = 0
    total_size: int = 0  # total number of elements across all tensors
    max_rank: int = 0  # maximum tensor rank
    max_dim: int = 0  # maximum dimension size
    estimated_flops: float = 0.0
    is_1d: bool = False
    is_2d: bool = False
    has_hyper_edges: bool = False


@dataclass
class RoutingDecision:
    """Result of the routing analysis."""
    backend: Backend
    reason: RoutingReason
    profile: NetworkProfile
    confidence: float = 1.0
    alternatives: list[Backend] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def profile_tensor_network(tn) -> NetworkProfile:
    """Analyze a quimb TensorNetwork and return its structural profile.

    Parameters
    ----------
    tn : quimb.tensor.TensorNetwork
        The tensor network to analyze.

    Returns
    -------
    NetworkProfile
        Structural characteristics for routing decisions.
    """
    tensors = list(tn.tensors)
    num_tensors = len(tensors)

    if num_tensors == 0:
        return NetworkProfile()

    total_size = sum(t.size for t in tensors)
    max_rank = max(len(t.shape) for t in tensors)
    max_dim = max(max(t.shape) if t.shape else 0 for t in tensors)
    num_indices = len(tn.ind_map)

    return NetworkProfile(
        num_tensors=num_tensors,
        num_indices=num_indices,
        total_size=total_size,
        max_rank=max_rank,
        max_dim=max_dim,
    )


def route(tn, available_backends=None) -> RoutingDecision:
    """Decide which backend to use for contracting a tensor network.

    Parameters
    ----------
    tn : quimb.tensor.TensorNetwork
        The tensor network to route.
    available_backends : list[Backend], optional
        Which backends are installed and available.

    Returns
    -------
    RoutingDecision
        The recommended backend and reasoning.
    """
    if available_backends is None:
        available_backends = _detect_backends()

    profile = profile_tensor_network(tn)

    # Simple heuristic routing (v0.1)
    # Will be replaced with learned cost models
    if profile.total_size < 1_000_000:
        # Small networks — CPU is faster (no transfer overhead)
        return RoutingDecision(
            backend=Backend.NUMPY,
            reason=RoutingReason.NETWORK_TOO_SMALL,
            profile=profile,
            alternatives=[b for b in available_backends if b != Backend.NUMPY],
        )

    if Backend.CUTENSORNET in available_backends:
        return RoutingDecision(
            backend=Backend.CUTENSORNET,
            reason=RoutingReason.NETWORK_LARGE,
            profile=profile,
            confidence=0.8,
            alternatives=[Backend.CUPY, Backend.NUMPY],
        )

    if Backend.CUPY in available_backends:
        return RoutingDecision(
            backend=Backend.CUPY,
            reason=RoutingReason.NETWORK_LARGE,
            profile=profile,
            confidence=0.7,
            alternatives=[Backend.NUMPY],
        )

    # Fallback to numpy
    return RoutingDecision(
        backend=Backend.NUMPY,
        reason=RoutingReason.BACKEND_UNAVAILABLE,
        profile=profile,
    )


def _detect_backends() -> list[Backend]:
    """Detect which backends are available."""
    backends = [Backend.NUMPY]  # always available

    try:
        import cupy  # noqa: F401
        backends.append(Backend.CUPY)
    except ImportError:
        pass

    try:
        import cuquantum  # noqa: F401
        backends.append(Backend.CUTENSORNET)
    except ImportError:
        pass

    try:
        import jax  # noqa: F401
        backends.append(Backend.JAX)
    except ImportError:
        pass

    try:
        import torch  # noqa: F401
        backends.append(Backend.TORCH)
    except ImportError:
        pass

    return backends
