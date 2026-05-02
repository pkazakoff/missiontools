import numpy as np
import pytest

from missiontools import GroundStation, Spacecraft
from missiontools.orbit.access import earth_access_intervals

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_EPOCH = np.datetime64("2000-01-01T12:00:00", "us")
_T_END = _EPOCH + np.timedelta64(2 * 24 * 3600, "s")  # 2-day window

_SC_KW = dict(
    a=6_771_000.0,
    e=0.0006,
    i=np.radians(51.6),
    raan=np.radians(120.0),
    arg_p=np.radians(30.0),
    ma=np.radians(0.0),
    epoch=_EPOCH,
)

# Ground station roughly at London
_GS = GroundStation(lat=51.5, lon=-0.1)
_SC = Spacecraft(**_SC_KW)


# ===========================================================================
# Construction
# ===========================================================================


class TestGroundStationConstruct:
    def test_fields_stored(self):
        gs = GroundStation(lat=48.85, lon=2.35, alt=35.0)
        assert gs.lat == 48.85
        assert gs.lon == 2.35
        assert gs.alt == 35.0

    def test_default_alt_zero(self):
        gs = GroundStation(lat=0.0, lon=0.0)
        assert gs.alt == 0.0

    def test_invalid_lat_too_large(self):
        with pytest.raises(ValueError, match="lat"):
            GroundStation(lat=91.0, lon=0.0)

    def test_invalid_lat_too_small(self):
        with pytest.raises(ValueError, match="lat"):
            GroundStation(lat=-91.0, lon=0.0)

    def test_boundary_lats_accepted(self):
        GroundStation(lat=90.0, lon=0.0)
        GroundStation(lat=-90.0, lon=0.0)

    def test_lon_out_of_range_accepted(self):
        """Any longitude is valid (trig is periodic)."""
        gs = GroundStation(lat=0.0, lon=270.0)
        assert gs.lon == 270.0


# ===========================================================================
# access() method
# ===========================================================================


class TestGroundStationAccess:
    def test_returns_list(self):
        result = _GS.access(_SC, _EPOCH, _T_END)
        assert isinstance(result, list)

    def test_intervals_are_tuples_of_datetime64(self):
        result = _GS.access(_SC, _EPOCH, _T_END)
        assert len(result) > 0
        for start, end in result:
            assert start.dtype == np.dtype("datetime64[us]")
            assert end.dtype == np.dtype("datetime64[us]")

    def test_intervals_are_ordered(self):
        """Each window must have start < end and windows must not overlap."""
        result = _GS.access(_SC, _EPOCH, _T_END)
        assert len(result) > 0
        for start, end in result:
            assert start < end
        for i in range(len(result) - 1):
            assert result[i][1] <= result[i + 1][0]

    def test_at_least_one_pass_in_two_days(self):
        """ISS-like orbit visible from London at least once over 2 days."""
        result = _GS.access(_SC, _EPOCH, _T_END)
        assert len(result) >= 1

    def test_no_access_from_south_pole_low_incl(self):
        """A low-inclination orbit (5°) should never be visible from the South Pole."""
        sc_low = Spacecraft(
            a=6_771_000.0,
            e=0.0,
            i=np.radians(5.0),
            raan=0.0,
            arg_p=0.0,
            ma=0.0,
            epoch=_EPOCH,
        )
        gs_pole = GroundStation(lat=-90.0, lon=0.0)
        result = gs_pole.access(sc_low, _EPOCH, _T_END)
        assert result == []

    def test_higher_el_min_fewer_or_equal_passes(self):
        passes_0 = _GS.access(_SC, _EPOCH, _T_END, el_min_deg=0.0)
        passes_30 = _GS.access(_SC, _EPOCH, _T_END, el_min_deg=30.0)
        assert len(passes_30) <= len(passes_0)

    def test_matches_earth_access_intervals_directly(self):
        """gs.access() must return exactly the same result as earth_access_intervals."""
        expected = earth_access_intervals(
            t_start=_EPOCH,
            t_end=_T_END,
            keplerian_params=_SC.keplerian_params,
            lat=np.radians(_GS.lat),
            lon=np.radians(_GS.lon),
            alt=_GS.alt,
            el_min=np.radians(5.0),
            propagator_type=_SC.propagator_type,
            max_step=np.timedelta64(30, "s"),
        )
        result = _GS.access(
            _SC, _EPOCH, _T_END, el_min_deg=5.0, max_step=np.timedelta64(30, "s")
        )
        assert len(result) == len(expected)
        for (rs, re), (es, ee) in zip(result, expected):
            assert rs == es
            assert re == ee

    def test_custom_step_accepted(self):
        result = _GS.access(_SC, _EPOCH, _T_END, max_step=np.timedelta64(60, "s"))
        assert isinstance(result, list)

    def test_j2_propagator_via_spacecraft(self):
        sc_j2 = Spacecraft(**_SC_KW, propagator_type="j2")
        result = _GS.access(sc_j2, _EPOCH, _T_END)
        assert isinstance(result, list)
        assert len(result) >= 1
