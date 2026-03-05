"""Size-aware LRU cache for expensive computations.

Provides a memory-budget-based cache that evicts least-recently-used entries
when total stored array memory exceeds a configurable limit, rather than
evicting by entry count like :func:`functools.lru_cache`.

The primary use case is caching :func:`~missiontools.orbit.propagation.propagate_analytical`
results so that multi-sensor coverage analyses sharing the same spacecraft
orbit avoid redundant propagation.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import numpy.typing as npt

from .orbit.constants import EARTH_MU, EARTH_J2

# Default memory budget: 256 MiB
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024


class SizeAwareLRU:
    """LRU cache that evicts based on total memory of stored numpy arrays.

    Parameters
    ----------
    max_bytes : int
        Maximum total memory (bytes) of cached array data before eviction.
    """

    def __init__(self, max_bytes: int = _DEFAULT_MAX_BYTES) -> None:
        self._max_bytes = max_bytes
        self._cache: OrderedDict[Any, tuple[int, Any]] = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @max_bytes.setter
    def max_bytes(self, value: int) -> None:
        with self._lock:
            self._max_bytes = value
            self._evict()

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, key: Any) -> Any | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key][1]
            self._misses += 1
            return None

    def put(self, key: Any, value: Any, nbytes: int) -> None:
        with self._lock:
            if key in self._cache:
                old_nbytes, _ = self._cache.pop(key)
                self._total_bytes -= old_nbytes
            self._cache[key] = (nbytes, value)
            self._total_bytes += nbytes
            self._cache.move_to_end(key)
            self._evict()

    def _evict(self) -> None:
        while self._total_bytes > self._max_bytes and self._cache:
            _, (nbytes, _) = self._cache.popitem(last=False)
            self._total_bytes -= nbytes

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._total_bytes = 0
            self._hits = 0
            self._misses = 0


def _make_propagation_key(
    t: npt.NDArray,
    epoch: np.datetime64,
    a: float,
    e: float,
    i: float,
    arg_p: float,
    raan: float,
    ma: float,
    type: str,
    central_body_mu: float,
    central_body_j2: float,
) -> tuple:
    """Build a hashable cache key for propagate_analytical."""
    t_hash = hashlib.sha256(np.asarray(t, dtype='datetime64[us]').tobytes()).digest()
    epoch_us = int(np.datetime64(epoch, 'us').view(np.int64))
    return (
        t_hash,
        epoch_us,
        float(a), float(e), float(i),
        float(arg_p), float(raan), float(ma),
        type,
        float(central_body_mu), float(central_body_j2),
    )


# Module-level singleton cache
_propagation_cache = SizeAwareLRU()


def cached_propagate_analytical(t, *, epoch, a, e, i, arg_p, raan, ma,
                                 type="twobody",
                                 central_body_mu=EARTH_MU,
                                 central_body_j2=EARTH_J2,
                                 central_body_radius=None,
                                 _cache: SizeAwareLRU = _propagation_cache,
                                 **_ignored):
    """Cache-aware wrapper around propagate_analytical.

    Checks the module-level propagation cache before calling
    :func:`~missiontools.orbit.propagation.propagate_analytical`.
    Results are stored keyed on orbital parameters + time array hash.

    Accepts and ignores ``central_body_radius`` for compatibility with
    ``keplerian_params`` dict unpacking.
    """
    from .orbit.propagation import propagate_analytical

    key = _make_propagation_key(t, epoch, a, e, i, arg_p, raan, ma,
                                type, central_body_mu, central_body_j2)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    r, v = propagate_analytical(t, epoch=epoch, a=a, e=e, i=i,
                                 arg_p=arg_p, raan=raan, ma=ma,
                                 type=type, central_body_mu=central_body_mu,
                                 central_body_j2=central_body_j2)
    nbytes = r.nbytes + v.nbytes
    _cache.put(key, (r, v), nbytes)
    return r, v


def clear_cache() -> None:
    """Clear the propagation cache."""
    _propagation_cache.clear()


def set_cache_limit(max_bytes: int) -> None:
    """Set the maximum memory budget for the propagation cache.

    Parameters
    ----------
    max_bytes : int
        Maximum total bytes of cached array data. Existing entries that
        exceed the new limit are evicted immediately (LRU order).
    """
    _propagation_cache.max_bytes = max_bytes


def cache_info() -> dict:
    """Return cache statistics.

    Returns
    -------
    dict
        Keys: ``hits``, ``misses``, ``entries``, ``total_bytes``, ``max_bytes``.
    """
    return {
        'hits': _propagation_cache.hits,
        'misses': _propagation_cache.misses,
        'entries': len(_propagation_cache),
        'total_bytes': _propagation_cache.total_bytes,
        'max_bytes': _propagation_cache.max_bytes,
    }
