import numpy as np
import pytest

from missiontools.cache import (
    SizeAwareLRU,
    cached_propagate_analytical,
    clear_cache,
    set_cache_limit,
    cache_info,
    _propagation_cache,
)
from missiontools.orbit.propagation import propagate_analytical
from missiontools.orbit.constants import EARTH_MU, EARTH_J2


# ---------------------------------------------------------------------------
# SizeAwareLRU unit tests
# ---------------------------------------------------------------------------

class TestSizeAwareLRU:

    def test_put_and_get(self):
        c = SizeAwareLRU(max_bytes=1024)
        c.put("k1", "v1", 100)
        assert c.get("k1") == "v1"
        assert len(c) == 1
        assert c.total_bytes == 100

    def test_miss_returns_none(self):
        c = SizeAwareLRU(max_bytes=1024)
        assert c.get("missing") is None

    def test_eviction_by_size(self):
        c = SizeAwareLRU(max_bytes=200)
        c.put("k1", "v1", 100)
        c.put("k2", "v2", 100)
        assert len(c) == 2
        # Adding k3 should evict k1 (LRU)
        c.put("k3", "v3", 100)
        assert c.get("k1") is None
        assert c.get("k2") == "v2"
        assert c.get("k3") == "v3"
        assert c.total_bytes == 200

    def test_lru_order(self):
        c = SizeAwareLRU(max_bytes=200)
        c.put("k1", "v1", 100)
        c.put("k2", "v2", 100)
        # Access k1 to make it recently used
        c.get("k1")
        # Adding k3 should evict k2 (now LRU)
        c.put("k3", "v3", 100)
        assert c.get("k1") == "v1"
        assert c.get("k2") is None
        assert c.get("k3") == "v3"

    def test_overwrite_key(self):
        c = SizeAwareLRU(max_bytes=1024)
        c.put("k1", "v1", 100)
        c.put("k1", "v2", 200)
        assert c.get("k1") == "v2"
        assert len(c) == 1
        assert c.total_bytes == 200

    def test_clear(self):
        c = SizeAwareLRU(max_bytes=1024)
        c.put("k1", "v1", 100)
        c.put("k2", "v2", 200)
        c.clear()
        assert len(c) == 0
        assert c.total_bytes == 0
        assert c.hits == 0
        assert c.misses == 0

    def test_hit_miss_counters(self):
        c = SizeAwareLRU(max_bytes=1024)
        c.put("k1", "v1", 100)
        c.get("k1")
        c.get("missing")
        assert c.hits == 1
        assert c.misses == 1

    def test_max_bytes_setter_triggers_eviction(self):
        c = SizeAwareLRU(max_bytes=1024)
        c.put("k1", "v1", 500)
        c.put("k2", "v2", 500)
        assert len(c) == 2
        c.max_bytes = 500
        assert len(c) == 1
        assert c.total_bytes == 500

    def test_large_item_evicts_everything(self):
        c = SizeAwareLRU(max_bytes=300)
        c.put("k1", "v1", 100)
        c.put("k2", "v2", 100)
        # Insert item larger than total of existing
        c.put("k3", "v3", 300)
        assert c.get("k1") is None
        assert c.get("k2") is None
        assert c.get("k3") == "v3"


# ---------------------------------------------------------------------------
# Propagation caching integration tests
# ---------------------------------------------------------------------------

# Standard LEO orbit params
_KP = dict(
    epoch=np.datetime64('2025-01-01T00:00:00', 'us'),
    a=6_878_000.0,
    e=0.001,
    i=np.radians(97.4),
    arg_p=np.radians(0.0),
    raan=np.radians(0.0),
    ma=np.radians(0.0),
    central_body_mu=EARTH_MU,
    central_body_j2=EARTH_J2,
)

_T = np.arange(
    np.datetime64('2025-01-01T00:00:00', 'us'),
    np.datetime64('2025-01-01T01:00:00', 'us'),
    np.timedelta64(60, 's'),
)


class TestCachedPropagation:

    def setup_method(self):
        clear_cache()

    def test_cached_matches_uncached(self):
        r_ref, v_ref = propagate_analytical(_T, **_KP, propagator_type='twobody')
        r_cached, v_cached = cached_propagate_analytical(
            _T, **_KP, propagator_type='twobody')
        np.testing.assert_array_equal(r_ref, r_cached)
        np.testing.assert_array_equal(v_ref, v_cached)

    def test_second_call_is_cache_hit(self):
        clear_cache()
        cached_propagate_analytical(_T, **_KP, propagator_type='twobody')
        info1 = cache_info()
        assert info1['misses'] == 1
        assert info1['hits'] == 0

        cached_propagate_analytical(_T, **_KP, propagator_type='twobody')
        info2 = cache_info()
        assert info2['hits'] == 1
        assert info2['misses'] == 1

    def test_different_times_are_different_keys(self):
        clear_cache()
        t2 = _T[:30]
        cached_propagate_analytical(_T, **_KP, propagator_type='twobody')
        cached_propagate_analytical(t2, **_KP, propagator_type='twobody')
        info = cache_info()
        assert info['misses'] == 2
        assert info['entries'] == 2

    def test_j2_cached_matches_uncached(self):
        r_ref, v_ref = propagate_analytical(_T, **_KP, propagator_type='j2')
        r_cached, v_cached = cached_propagate_analytical(
            _T, **_KP, propagator_type='j2')
        np.testing.assert_array_equal(r_ref, r_cached)
        np.testing.assert_array_equal(v_ref, v_cached)

    def test_cache_info_reports_bytes(self):
        clear_cache()
        cached_propagate_analytical(_T, **_KP, propagator_type='twobody')
        info = cache_info()
        # r and v are both (N, 3) float64 = N * 3 * 8 bytes each
        expected = 2 * len(_T) * 3 * 8
        assert info['total_bytes'] == expected

    def test_clear_cache(self):
        cached_propagate_analytical(_T, **_KP, propagator_type='twobody')
        clear_cache()
        info = cache_info()
        assert info['entries'] == 0
        assert info['total_bytes'] == 0

    def test_set_cache_limit(self):
        cached_propagate_analytical(_T, **_KP, propagator_type='twobody')
        set_cache_limit(1)  # 1 byte — should evict everything
        info = cache_info()
        assert info['entries'] == 0
        # Restore
        set_cache_limit(256 * 1024 * 1024)

    def test_keplerian_params_dict_unpacking(self):
        """cached_propagate_analytical should accept and ignore central_body_radius."""
        kp_with_radius = {**_KP, 'central_body_radius': 6_371_000.0}
        r, v = cached_propagate_analytical(_T, **kp_with_radius, propagator_type='twobody')
        assert r.shape == (len(_T), 3)
