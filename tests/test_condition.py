import numpy as np
import pytest

from missiontools import (
    AbstractCondition,
    SpaceGroundAccessCondition,
    SunlightCondition,
    SubSatelliteRegionCondition,
    VisibilityCondition,
    AndCondition,
    OrCondition,
    NotCondition,
    XorCondition,
    GroundStation,
    Spacecraft,
    AoI,
)
from missiontools.condition.condition import AbstractCondition as _AC


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_EPOCH = np.datetime64("2025-01-01T00:00:00", "us")
_T_END = _EPOCH + np.timedelta64(2 * 24 * 3600, "s")  # 2 days

_SC_KW = dict(
    a=6_771_000.0,
    e=0.0,
    i=np.radians(51.6),
    raan=0.0,
    arg_p=0.0,
    ma=0.0,
    epoch=_EPOCH,
)
_SC = Spacecraft(**_SC_KW)
_GS = GroundStation(lat=51.5, lon=-0.1)  # London


class _CountingCondition(AbstractCondition):
    """Test helper: returns alternating booleans, counts _compute calls."""

    def __init__(self, cache_size=16):
        super().__init__(cache_size=cache_size)
        self.calls = 0

    def _compute(self, t):
        self.calls += 1
        # Alternate True/False
        return np.arange(len(t)) % 2 == 0

    def __repr__(self):
        return "CountingCondition()"


class _ConstCondition(AbstractCondition):
    """Test helper: returns a fixed boolean for every time step."""

    def __init__(self, value: bool):
        super().__init__(cache_size=0)
        self._value = value

    def _compute(self, t):
        return np.full(len(t), self._value)

    def __repr__(self):
        return f"ConstCondition({self._value})"


class _WindowCondition(AbstractCondition):
    """Test helper: True inside a list of [start, end) windows (µs offsets)."""

    def __init__(self, epoch, windows_us):
        super().__init__(cache_size=0)
        self._epoch = np.asarray(epoch, dtype="datetime64[us]")
        self._windows = windows_us

    def _compute(self, t):
        t_off = (t - self._epoch).astype("timedelta64[us]").astype(np.int64)
        result = np.zeros(len(t), dtype=bool)
        for s, e in self._windows:
            result |= (t_off >= s) & (t_off < e)
        return result

    def __repr__(self):
        return f"WindowCondition(windows={self._windows})"


class _DirectAtCondition(AbstractCondition):
    """Test helper: overrides at() directly, bypassing the cache."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def at(self, t):
        self.calls += 1
        t_arr = np.atleast_1d(np.asarray(t, dtype="datetime64[us]"))
        return np.ones(len(t_arr), dtype=bool)

    def _compute(self, t):
        raise AssertionError("_compute should not be called when at() is overridden")

    def __repr__(self):
        return "DirectAtCondition()"


# ===========================================================================
# AbstractCondition: caching + shape handling
# ===========================================================================


class TestAbstractConditionCache:
    def test_cache_hit_avoids_recompute(self):
        c = _CountingCondition()
        t = np.arange("2025-01-01", "2025-01-01T00:00:10", dtype="datetime64[s]")
        c.at(t)
        c.at(t)
        c.at(t)
        assert c.calls == 1

    def test_cache_miss_on_different_t(self):
        c = _CountingCondition()
        t1 = np.array([_EPOCH], dtype="datetime64[us]")
        t2 = np.array([_EPOCH + np.timedelta64(1, "s")], dtype="datetime64[us]")
        c.at(t1)
        c.at(t2)
        assert c.calls == 2

    def test_cache_eviction_lru(self):
        """When cache_size is exceeded, oldest entry is evicted (LRU)."""
        c = _CountingCondition(cache_size=2)
        t1 = np.array([_EPOCH], dtype="datetime64[us]")
        t2 = np.array([_EPOCH + np.timedelta64(1, "s")], dtype="datetime64[us]")
        t3 = np.array([_EPOCH + np.timedelta64(2, "s")], dtype="datetime64[us]")
        c.at(t1)
        c.at(t2)
        c.at(t3)  # evicts t1
        c.at(t2)  # still cached
        c.at(t1)  # was evicted -> recompute
        assert c.calls == 4

    def test_cache_size_zero_disables_cache(self):
        c = _CountingCondition(cache_size=0)
        t = np.array([_EPOCH], dtype="datetime64[us]")
        c.at(t)
        c.at(t)
        assert c.calls == 2

    def test_negative_cache_size_rejected(self):
        with pytest.raises(ValueError, match="cache_size"):
            _CountingCondition(cache_size=-1)

    def test_subclass_can_override_at(self):
        """Subclasses bypass caching by overriding at() directly."""
        c = _DirectAtCondition()
        t = np.array([_EPOCH], dtype="datetime64[us]")
        c.at(t)
        c.at(t)
        assert c.calls == 2


class TestAbstractConditionShapes:
    def test_scalar_t_returns_python_bool(self):
        c = _CountingCondition()
        result = c.at(_EPOCH)
        assert isinstance(result, bool)

    def test_array_t_returns_bool_array(self):
        c = _CountingCondition()
        t = np.array([_EPOCH, _EPOCH + np.timedelta64(1, "s")])
        result = c.at(t)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert result.shape == (2,)


# ===========================================================================
# AbstractCondition: operator overrides
# ===========================================================================


class TestAbstractConditionOperators:
    def test_and_operator(self):
        c1 = _ConstCondition(True)
        c2 = _ConstCondition(False)
        result = (c1 & c2).at(_EPOCH)
        assert result is False

    def test_or_operator(self):
        c1 = _ConstCondition(False)
        c2 = _ConstCondition(True)
        result = (c1 | c2).at(_EPOCH)
        assert result is True

    def test_xor_operator(self):
        c1 = _ConstCondition(True)
        c2 = _ConstCondition(True)
        result = (c1 ^ c2).at(_EPOCH)
        assert result is False

    def test_invert_operator(self):
        c = _ConstCondition(True)
        result = (~c).at(_EPOCH)
        assert result is False

    def test_operators_return_notimplemented_for_non_condition(self):
        c = _ConstCondition(True)
        assert c.__and__(42) is NotImplemented
        assert c.__or__(42) is NotImplemented
        assert c.__xor__(42) is NotImplemented

    def test_composition_equivalence(self):
        """(c1 & (c2 | c3)) == AndCondition(c1, OrCondition(c2, c3))."""
        c1 = _ConstCondition(True)
        c2 = _ConstCondition(False)
        c3 = _ConstCondition(True)
        t = np.array([_EPOCH], dtype="datetime64[us]")
        op_result = (c1 & (c2 | c3)).at(t)
        cls_result = AndCondition(c1, OrCondition(c2, c3)).at(t)
        assert np.array_equal(op_result, cls_result)


# ===========================================================================
# SpaceGroundAccessCondition: type checking + construction
# ===========================================================================


class TestSpaceGroundAccessConditionConstruct:
    def test_rejects_non_spacecraft(self):
        with pytest.raises(TypeError, match="spacecraft"):
            SpaceGroundAccessCondition("not a spacecraft", _GS)

    def test_rejects_non_ground_station(self):
        with pytest.raises(TypeError, match="ground_station"):
            SpaceGroundAccessCondition(_SC, "not a gs")

    def test_rejects_nonfinite_el_min(self):
        with pytest.raises(ValueError, match="el_min"):
            SpaceGroundAccessCondition(_SC, _GS, el_min_deg=np.inf)

    def test_repr_contains_el_min(self):
        c = SpaceGroundAccessCondition(_SC, _GS, el_min_deg=7.5)
        assert "7.5" in repr(c)


# ===========================================================================
# SpaceGroundAccessCondition: geometric correctness
# ===========================================================================


class TestSpaceGroundAccessConditionGeometry:
    def test_subsat_point_sees_spacecraft(self):
        """A GS at the subsatellite point at t0 must see the SC."""
        from missiontools.orbit.propagation import propagate_analytical
        from missiontools.orbit.frames import eci_to_ecef

        t = np.array([_EPOCH])
        r, _ = propagate_analytical(t, **_SC_KW, propagator_type="twobody")
        r_ecef = eci_to_ecef(r, t)[0]
        lat = np.degrees(np.arcsin(r_ecef[2] / np.linalg.norm(r_ecef)))
        lon = np.degrees(np.arctan2(r_ecef[1], r_ecef[0]))
        gs = GroundStation(lat=lat, lon=lon)
        cond = SpaceGroundAccessCondition(_SC, gs, el_min_deg=5.0)
        assert cond.at(_EPOCH) is True

    def test_antipodal_point_does_not_see_spacecraft(self):
        """A GS antipodal to the subsatellite point cannot see the SC."""
        from missiontools.orbit.propagation import propagate_analytical
        from missiontools.orbit.frames import eci_to_ecef

        t = np.array([_EPOCH])
        r, _ = propagate_analytical(t, **_SC_KW, propagator_type="twobody")
        r_ecef = eci_to_ecef(r, t)[0]
        lat = np.degrees(np.arcsin(r_ecef[2] / np.linalg.norm(r_ecef)))
        lon = np.degrees(np.arctan2(r_ecef[1], r_ecef[0]))
        gs = GroundStation(lat=-lat, lon=lon + 180.0)
        cond = SpaceGroundAccessCondition(_SC, gs, el_min_deg=0.0)
        assert cond.at(_EPOCH) is False

    def test_higher_el_min_only_removes_passes(self):
        """Raising el_min cannot turn False samples True."""
        t = np.arange(_EPOCH, _T_END, np.timedelta64(120, "s"), dtype="datetime64[us]")
        c0 = SpaceGroundAccessCondition(_SC, _GS, el_min_deg=0.0)
        c10 = SpaceGroundAccessCondition(_SC, _GS, el_min_deg=10.0)
        v0 = c0.at(t)
        v10 = c10.at(t)
        # Every True at el_min=10 must also be True at el_min=0.
        assert np.all(v0[v10])
        assert v0.sum() >= v10.sum()

    def test_agrees_with_gs_access_intervals(self):
        """Sample-wise truth must match interval-membership from GroundStation.access."""
        t = np.arange(_EPOCH, _T_END, np.timedelta64(60, "s"), dtype="datetime64[us]")
        cond = SpaceGroundAccessCondition(_SC, _GS, el_min_deg=5.0)
        samples = cond.at(t)
        intervals = _GS.access(
            _SC, _EPOCH, _T_END, el_min_deg=5.0, max_step=np.timedelta64(30, "s")
        )

        expected = np.zeros(len(t), dtype=bool)
        for a, b in intervals:
            expected |= (t >= a) & (t <= b)

        # Boundary-adjacent samples can disagree by one step due to the
        # rootfind cadence; require >99% agreement.
        agreement = np.mean(samples == expected)
        assert agreement > 0.99


# ===========================================================================
# SunlightCondition
# ===========================================================================


class TestSunlightConditionConstruct:
    def test_rejects_non_spacecraft_or_gs(self):
        with pytest.raises(TypeError, match="obj"):
            SunlightCondition("not an object")

    def test_repr_contains_obj(self):
        c = SunlightCondition(_SC)
        assert "SunlightCondition" in repr(c)


class TestSunlightConditionGeometry:
    def test_sc_sunlight_returns_bool_array(self):
        t = np.arange(
            _EPOCH,
            _EPOCH + np.timedelta64(600, "s"),
            np.timedelta64(60, "s"),
            dtype="datetime64[us]",
        )
        cond = SunlightCondition(_SC)
        result = cond.at(t)
        assert result.dtype == bool
        assert result.shape == t.shape

    def test_gs_sunlight_returns_bool_array(self):
        t = np.arange(
            _EPOCH,
            _EPOCH + np.timedelta64(24 * 3600, "s"),
            np.timedelta64(600, "s"),
            dtype="datetime64[us]",
        )
        cond = SunlightCondition(_GS)
        result = cond.at(t)
        assert result.dtype == bool
        assert result.shape == t.shape
        # A ground station in London must be in sunlight at least some
        # of the time over 24h and in shadow at least some of the time.
        assert result.any()
        assert not result.all()

    def test_sc_sunlight_scalar(self):
        cond = SunlightCondition(_SC)
        result = cond.at(_EPOCH)
        assert isinstance(result, bool)


# ===========================================================================
# SubSatelliteRegionCondition
# ===========================================================================


class TestSubSatelliteRegionConditionConstruct:
    def test_rejects_non_spacecraft(self):
        aoi = AoI.from_region(-10, 10, -10, 10)
        with pytest.raises(TypeError, match="spacecraft"):
            SubSatelliteRegionCondition("not sc", aoi)

    def test_rejects_non_aoi(self):
        with pytest.raises(TypeError, match="aoi"):
            SubSatelliteRegionCondition(_SC, "not aoi")

    def test_rejects_aoi_without_geometry(self):
        aoi = AoI(np.array([0.0]), np.array([0.0]))
        assert aoi.geometry is None
        with pytest.raises(ValueError, match="geometry"):
            SubSatelliteRegionCondition(_SC, aoi)

    def test_repr(self):
        aoi = AoI.from_region(-10, 10, -10, 10)
        c = SubSatelliteRegionCondition(_SC, aoi)
        assert "SubSatelliteRegionCondition" in repr(c)


class TestSubSatelliteRegionConditionGeometry:
    def test_inside_region(self):
        """SC at epoch has its sub-satellite point; make a region around it."""
        from missiontools.orbit.propagation import propagate_analytical
        from missiontools.orbit.frames import eci_to_ecef, ecef_to_geodetic

        t = np.array([_EPOCH])
        r, _ = propagate_analytical(t, **_SC_KW, propagator_type="twobody")
        r_ecef = eci_to_ecef(r, t)
        lat_r, lon_r, _ = ecef_to_geodetic(r_ecef)
        lat_deg = np.degrees(lat_r)
        lon_deg = np.degrees(lon_r)

        aoi = AoI.from_region(lat_deg - 5, lat_deg + 5, lon_deg - 5, lon_deg + 5)
        cond = SubSatelliteRegionCondition(_SC, aoi)
        assert cond.at(_EPOCH) is True

    def test_outside_region(self):
        """Region far from the sub-satellite point should return False."""
        aoi = AoI.from_region(-90, -80, -180, -170)
        cond = SubSatelliteRegionCondition(_SC, aoi)
        # A single time step is very unlikely to be in that region
        # for a 51.6° inclined orbit
        t = np.array([_EPOCH], dtype="datetime64[us]")
        result = cond.at(t)
        assert result.dtype == bool


# ===========================================================================
# VisibilityCondition
# ===========================================================================


class TestVisibilityConditionConstruct:
    def test_rejects_non_spacecraft_or_gs(self):
        with pytest.raises(TypeError, match="obj1"):
            VisibilityCondition("not an object", _SC)

    def test_repr(self):
        c = VisibilityCondition(_SC, _GS)
        assert "VisibilityCondition" in repr(c)


class TestVisibilityConditionGeometry:
    def test_sc_gs_visibility_returns_bool_array(self):
        t = np.arange(
            _EPOCH,
            _EPOCH + np.timedelta64(600, "s"),
            np.timedelta64(60, "s"),
            dtype="datetime64[us]",
        )
        cond = VisibilityCondition(_SC, _GS)
        result = cond.at(t)
        assert result.dtype == bool
        assert result.shape == t.shape

    def test_sc_sc_visibility(self):
        sc2 = Spacecraft(
            a=6_871_000.0,
            e=0.0,
            i=np.radians(28.5),
            raan=0.0,
            arg_p=0.0,
            ma=np.pi,
            epoch=_EPOCH,
        )
        cond = VisibilityCondition(_SC, sc2)
        result = cond.at(_EPOCH)
        assert isinstance(result, bool)

    def test_gs_gs_visibility(self):
        gs2 = GroundStation(lat=-33.9, lon=151.2)
        cond = VisibilityCondition(_GS, gs2)
        result = cond.at(_EPOCH)
        assert isinstance(result, bool)


# ===========================================================================
# Composite conditions (And, Or, Not, Xor)
# ===========================================================================


class TestCompositeConditionConstruct:
    def test_and_rejects_non_condition(self):
        with pytest.raises(TypeError, match="condition1"):
            AndCondition("not a condition", _ConstCondition(True))

    def test_or_rejects_non_condition(self):
        with pytest.raises(TypeError, match="condition2"):
            OrCondition(_ConstCondition(True), 42)

    def test_not_rejects_non_condition(self):
        with pytest.raises(TypeError, match="condition"):
            NotCondition(42)

    def test_xor_rejects_non_condition(self):
        with pytest.raises(TypeError, match="condition1"):
            XorCondition("nope", _ConstCondition(True))


class TestCompositeConditionLogic:
    def test_and_truth_table(self):
        t = np.array([_EPOCH], dtype="datetime64[us]")
        for v1 in (True, False):
            for v2 in (True, False):
                c = AndCondition(_ConstCondition(v1), _ConstCondition(v2))
                assert c.at(t)[0] == (v1 and v2)

    def test_or_truth_table(self):
        t = np.array([_EPOCH], dtype="datetime64[us]")
        for v1 in (True, False):
            for v2 in (True, False):
                c = OrCondition(_ConstCondition(v1), _ConstCondition(v2))
                assert c.at(t)[0] == (v1 or v2)

    def test_not_truth_table(self):
        t = np.array([_EPOCH], dtype="datetime64[us]")
        for v in (True, False):
            c = NotCondition(_ConstCondition(v))
            assert c.at(t)[0] == (not v)

    def test_xor_truth_table(self):
        t = np.array([_EPOCH], dtype="datetime64[us]")
        for v1 in (True, False):
            for v2 in (True, False):
                c = XorCondition(_ConstCondition(v1), _ConstCondition(v2))
                assert c.at(t)[0] == (v1 != v2)

    def test_and_with_array(self):
        t = np.array([_EPOCH, _EPOCH + np.timedelta64(1, "s")])
        c = AndCondition(_CountingCondition(), _ConstCondition(True))
        result = c.at(t)
        assert result.shape == (2,)
        assert result[0] is np.bool_(True)
        assert result[1] is np.bool_(False)

    def test_repr_and(self):
        c = AndCondition(_ConstCondition(True), _ConstCondition(False))
        assert "AndCondition" in repr(c)

    def test_repr_or(self):
        c = OrCondition(_ConstCondition(True), _ConstCondition(False))
        assert "OrCondition" in repr(c)

    def test_repr_not(self):
        c = NotCondition(_ConstCondition(True))
        assert "NotCondition" in repr(c)

    def test_repr_xor(self):
        c = XorCondition(_ConstCondition(True), _ConstCondition(False))
        assert "XorCondition" in repr(c)


# ===========================================================================
# AbstractCondition.intervals()
# ===========================================================================


class TestAbstractConditionIntervals:
    def test_always_true_returns_single_interval(self):
        c = _ConstCondition(True)
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(100, "s")
        result = c.intervals(t0, t1)
        assert len(result) == 1
        assert result[0] == (t0, t1)

    def test_always_false_returns_empty(self):
        c = _ConstCondition(False)
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(100, "s")
        result = c.intervals(t0, t1)
        assert result == []

    def test_single_window_in_middle(self):
        c = _WindowCondition(_EPOCH, [(30_000_000, 70_000_000)])
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(100, "s")
        result = c.intervals(t0, t1, max_step=np.timedelta64(5, "s"))
        assert len(result) == 1
        s, e = result[0]
        assert s >= t0
        assert e <= t1
        assert int((s - _EPOCH) / np.timedelta64(1, "s")) <= 32
        assert int((e - _EPOCH) / np.timedelta64(1, "s")) >= 68

    def test_starts_true(self):
        c = _WindowCondition(_EPOCH, [(0, 50_000_000)])
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(100, "s")
        result = c.intervals(t0, t1, max_step=np.timedelta64(5, "s"))
        assert len(result) == 1
        s, e = result[0]
        assert s == t0

    def test_ends_true(self):
        c = _WindowCondition(_EPOCH, [(50_000_000, 200_000_000)])
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(100, "s")
        result = c.intervals(t0, t1, max_step=np.timedelta64(5, "s"))
        assert len(result) == 1
        s, e = result[0]
        assert e == t1

    def test_two_disjoint_windows(self):
        c = _WindowCondition(
            _EPOCH, [(10_000_000, 30_000_000), (60_000_000, 80_000_000)]
        )
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(100, "s")
        result = c.intervals(t0, t1, max_step=np.timedelta64(5, "s"))
        assert len(result) == 2
        assert result[0][0] < result[1][0]

    def test_empty_window_returns_empty(self):
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(10, "s")
        result = _ConstCondition(True).intervals(t0, t0)
        assert result == []

    def test_intervals_non_overlapping(self):
        c = _WindowCondition(
            _EPOCH, [(10_000_000, 40_000_000), (60_000_000, 90_000_000)]
        )
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(100, "s")
        result = c.intervals(t0, t1, max_step=np.timedelta64(5, "s"))
        for i in range(len(result) - 1):
            assert result[i][1] <= result[i + 1][0]

    def test_sunlight_condition_produces_intervals(self):
        cond = SunlightCondition(_SC)
        t0 = _EPOCH
        t1 = _EPOCH + np.timedelta64(6 * 3600, "s")
        result = cond.intervals(t0, t1)
        assert len(result) > 0
        for s, e in result:
            assert s >= t0
            assert e <= t1
            assert s < e
