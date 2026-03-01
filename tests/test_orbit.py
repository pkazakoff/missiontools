import numpy as np
import pytest
from datetime import datetime, timedelta

from missiontools.orbit.propagation import (
    propagate_analytical, sun_synchronous_inclination, sun_synchronous_orbit,
    geostationary_orbit, highly_elliptical_orbit,
)
from missiontools.orbit.frames import eci_to_ecef
from missiontools.orbit.constants import EARTH_MU, EARTH_J2, EARTH_SEMI_MAJOR_AXIS

EPOCH = datetime(2025, 1, 1, 12, 0, 0)


def make_times(n, dt_seconds):
    return [EPOCH + timedelta(seconds=k * dt_seconds) for k in range(n)]


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# A generic inclined elliptical orbit used by several tests
ELLIPTIC = dict(
    a=8_000_000.0,
    e=0.3,
    i=np.radians(45.0),
    arg_p=np.radians(90.0),
    raan=np.radians(60.0),
    ma=0.5,
)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_shape():
    r, v = propagate_analytical(make_times(20, 60.0), EPOCH, **ELLIPTIC)
    assert r.shape == (20, 3)
    assert v.shape == (20, 3)


def test_single_timestep_shape():
    r, v = propagate_analytical([EPOCH], EPOCH, **ELLIPTIC)
    assert r.shape == (1, 3)
    assert v.shape == (1, 3)


# ---------------------------------------------------------------------------
# Circular orbit: radius is exactly a
# ---------------------------------------------------------------------------

def test_circular_orbit_constant_radius():
    """For e=0, |r| must equal a at every timestep (rotation preserves norms)."""
    a = 7_000_000.0
    r, _ = propagate_analytical(
        make_times(100, 60.0), EPOCH,
        a=a, e=0.0, i=np.radians(28.5),
        arg_p=0.0, raan=0.0, ma=0.0,
    )
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), a, rtol=1e-12)


# ---------------------------------------------------------------------------
# Vis-viva / specific orbital energy conservation
# ---------------------------------------------------------------------------

def test_energy_conservation():
    """ε = |v|²/2 − μ/|r| = −μ/(2a) must hold at every timestep."""
    a = ELLIPTIC["a"]
    r, v = propagate_analytical(make_times(200, 30.0), EPOCH, **ELLIPTIC)
    energy = 0.5 * np.sum(v**2, axis=1) - EARTH_MU / np.linalg.norm(r, axis=1)
    np.testing.assert_allclose(energy, -EARTH_MU / (2 * a), rtol=1e-10)


# ---------------------------------------------------------------------------
# Angular momentum conservation
# ---------------------------------------------------------------------------

def test_angular_momentum_conservation():
    """|r × v| = sqrt(μ · a · (1−e²)) must hold at every timestep."""
    a, e = ELLIPTIC["a"], ELLIPTIC["e"]
    r, v = propagate_analytical(make_times(200, 30.0), EPOCH, **ELLIPTIC)
    h_mag = np.linalg.norm(np.cross(r, v), axis=1)
    np.testing.assert_allclose(h_mag, np.sqrt(EARTH_MU * a * (1 - e**2)), rtol=1e-10)


# ---------------------------------------------------------------------------
# Orbit closes after one period
# ---------------------------------------------------------------------------

def test_orbit_closes_after_one_period():
    """After exactly one orbital period the state vector must repeat."""
    a, e = ELLIPTIC["a"], ELLIPTIC["e"]
    T = 2 * np.pi * np.sqrt(a**3 / EARTH_MU)
    times = [EPOCH, EPOCH + timedelta(seconds=T)]
    r, v = propagate_analytical(times, EPOCH, **ELLIPTIC)
    np.testing.assert_allclose(r[0], r[1], atol=1e-3)   # 1 mm
    np.testing.assert_allclose(v[0], v[1], atol=1e-6)   # 1 µm/s


# ---------------------------------------------------------------------------
# Geometry checks
# ---------------------------------------------------------------------------

def test_perigee_position_along_eci_x():
    """At epoch with ma=0, i=0, raan=0, arg_p=0 the satellite is at
    perigee on the +x axis: r = (a(1−e), 0, 0)."""
    a, e = 8_000_000.0, 0.2
    r, _ = propagate_analytical(
        [EPOCH], EPOCH,
        a=a, e=e, i=0.0, arg_p=0.0, raan=0.0, ma=0.0,
    )
    np.testing.assert_allclose(r[0], [a * (1 - e), 0.0, 0.0], atol=1e-3)


def test_equatorial_orbit_zero_z():
    """For i=0 the z-component of every position vector must be zero."""
    r, _ = propagate_analytical(
        make_times(50, 120.0), EPOCH,
        a=7_000_000.0, e=0.05, i=0.0,
        arg_p=np.radians(45.0), raan=0.0, ma=0.5,
    )
    np.testing.assert_allclose(r[:, 2], 0.0, atol=1e-6)


def test_polar_orbit_crosses_z_axis():
    """For i=π/2 the angular momentum vector must lie in the xy-plane
    (h_z ≈ 0), i.e. the orbit passes through both poles."""
    r, v = propagate_analytical(
        make_times(100, 60.0), EPOCH,
        a=7_000_000.0, e=0.01, i=np.pi / 2,
        arg_p=0.0, raan=0.0, ma=0.0,
    )
    h = np.cross(r, v)
    np.testing.assert_allclose(h[:, 2], 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

_BASE = dict(
    epoch=EPOCH,
    a=7_000_000.0, e=0.01,
    i=np.radians(45.0),
    arg_p=0.5, raan=0.5, ma=0.5,
)


@pytest.mark.parametrize("override,match", [
    ({"a": 0.0},               "Semi-major axis"),
    ({"a": -1.0},              "Semi-major axis"),
    ({"e": -0.1},              "Eccentricity"),
    ({"e": 1.0},               "Eccentricity"),
    ({"i": -0.1},              "Inclination"),
    ({"i": np.pi + 0.1},       "Inclination"),
    ({"arg_p": -0.1},          "arg_p"),
    ({"arg_p": 2 * np.pi},     "arg_p"),
    ({"raan": -0.1},           "raan"),
    ({"raan": 2 * np.pi},      "raan"),
    ({"ma": -0.1},             "ma"),
    ({"ma": 2 * np.pi},        "ma"),
    ({"central_body_mu": 0.0}, "central_body_mu"),
    ({"central_body_mu":-1.0}, "central_body_mu"),
])
def test_input_validation_raises(override, match):
    kwargs = {**_BASE, **override}
    with pytest.raises(ValueError, match=match):
        propagate_analytical([EPOCH], **kwargs)


# ===========================================================================
# sun_synchronous_inclination
# ===========================================================================

# Mean solar rate and J2 constants used to cross-check formulas
_N_SUN = 2.0 * np.pi / (365.25 * 86400.0)   # rad/s


# ---------------------------------------------------------------------------
# Return value is in radians and in the retrograde band (π/2, π)
# ---------------------------------------------------------------------------

def test_sso_inclination_retrograde():
    """SSO inclination must be in (90°, 180°) for any valid Earth orbit."""
    i = sun_synchronous_inclination(7_000_000.0, 0.0)
    assert np.pi / 2 < i < np.pi


# ---------------------------------------------------------------------------
# Circular ISS-altitude orbit → ~97.4° (well-known reference value)
# ---------------------------------------------------------------------------

def test_sso_known_value_iss_altitude():
    """At ~400 km altitude (a ≈ 6 771 km), SSO inclination ≈ 97.0°."""
    i = sun_synchronous_inclination(6_771_000.0)
    assert abs(np.degrees(i) - 97.0) < 0.1


# ---------------------------------------------------------------------------
# Higher orbit → larger inclination (closer to 180°)
# ---------------------------------------------------------------------------

def test_sso_inclination_increases_with_altitude():
    """SSO inclination increases monotonically with semi-major axis."""
    i_low  = sun_synchronous_inclination(6_600_000.0)
    i_mid  = sun_synchronous_inclination(7_200_000.0)
    i_high = sun_synchronous_inclination(8_000_000.0)
    assert i_low < i_mid < i_high


# ---------------------------------------------------------------------------
# Round-trip: computed i reproduces the target RAAN drift
# ---------------------------------------------------------------------------

def test_sso_raan_drift_matches_solar():
    """Plugging the returned inclination back into the J2 drift formula must
    give a drift rate equal to the mean solar rate within 0.01 %."""
    a, e = 7_100_000.0, 0.001
    i = sun_synchronous_inclination(a, e)

    n     = np.sqrt(EARTH_MU / a**3)
    p     = a * (1.0 - e**2)
    j2_r2 = EARTH_J2 / EARTH_MU
    raan_dot = (-3.0 * n * j2_r2 / (2.0 * p**2)) * np.cos(i)

    assert abs(raan_dot / _N_SUN - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Non-zero eccentricity is handled correctly
# ---------------------------------------------------------------------------

def test_sso_eccentric_orbit():
    """Eccentric orbit (e = 0.05) must return a valid inclination."""
    i = sun_synchronous_inclination(7_500_000.0, e=0.05)
    assert np.pi / 2 < i < np.pi


# ---------------------------------------------------------------------------
# Return type is plain float
# ---------------------------------------------------------------------------

def test_sso_return_type():
    """Function must return a Python float."""
    i = sun_synchronous_inclination(7_000_000.0)
    assert isinstance(i, float)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,match", [
    ({"a": 0.0},                    "Semi-major axis"),
    ({"a": -1.0},                   "Semi-major axis"),
    ({"a": 7e6, "e": -0.1},         "Eccentricity"),
    ({"a": 7e6, "e": 1.0},          "Eccentricity"),
    ({"a": 7e6, "central_body_mu": 0.0},  "central_body_mu"),
    ({"a": 7e6, "central_body_mu": -1.0}, "central_body_mu"),
])
def test_sso_validation_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sun_synchronous_inclination(**kwargs)


def test_sso_no_solution_raises():
    """An orbit far too large or with tiny j2 should have |cos i| > 1."""
    with pytest.raises(ValueError, match="No sun-synchronous orbit"):
        # Very large a → tiny n → cos_i >> 1
        sun_synchronous_inclination(1e9, 0.0)


# ===========================================================================
# sun_synchronous_orbit
# ===========================================================================

_SSO_ALT  = 500_000.0   # 500 km altitude
_SSO_LTAN = "10:30"     # Local time of ascending node
_SSO_EPOCH = np.datetime64('2024-06-21T12:00:00', 'us')   # near summer solstice


def _sso():
    return sun_synchronous_orbit(_SSO_ALT, _SSO_LTAN, epoch=_SSO_EPOCH)


# ---------------------------------------------------------------------------
# Dict keys match propagate_analytical parameters
# ---------------------------------------------------------------------------

def test_sso_orbit_dict_keys():
    """Returned dict must contain exactly the required propagate_analytical keys."""
    required = {'epoch', 'a', 'e', 'i', 'arg_p', 'raan', 'ma'}
    result = _sso()
    assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# Circular orbit (e = 0, arg_p = 0)
# ---------------------------------------------------------------------------

def test_sso_orbit_circular():
    """SSO orbit must be circular: e == 0 and arg_p == 0."""
    r = _sso()
    assert r['e'] == 0.0
    assert r['arg_p'] == 0.0
    assert r['ma'] == 0.0


# ---------------------------------------------------------------------------
# Semi-major axis matches altitude + equatorial radius
# ---------------------------------------------------------------------------

def test_sso_orbit_semi_major_axis():
    """a must equal EARTH_SEMI_MAJOR_AXIS + altitude."""
    r = _sso()
    assert r['a'] == pytest.approx(EARTH_SEMI_MAJOR_AXIS + _SSO_ALT)


# ---------------------------------------------------------------------------
# Inclination matches sun_synchronous_inclination
# ---------------------------------------------------------------------------

def test_sso_orbit_inclination():
    """i must match the value returned by sun_synchronous_inclination."""
    r = _sso()
    i_ref = sun_synchronous_inclination(r['a'])
    assert r['i'] == pytest.approx(i_ref)


# ---------------------------------------------------------------------------
# RAAN reproduces the requested LTAN
# ---------------------------------------------------------------------------

def test_sso_orbit_ltan_ascending():
    """Computing LTAN from RAAN and Sun's RA at epoch must match the input."""
    r = _sso()

    # Recompute Sun's RA at the epoch (same algorithm as the function)
    d = float((np.asarray(_SSO_EPOCH, dtype='datetime64[us]') -
               np.datetime64('2000-01-01T12:00:00', 'us')).astype(np.float64)
              ) * 1e-6 / 86400.0
    L_deg = (280.460 + 0.9856474 * d) % 360.0
    g_rad = np.radians((357.528 + 0.9856003 * d) % 360.0)
    lam   = np.radians(L_deg) + np.radians(1.915 * np.sin(g_rad) + 0.020 * np.sin(2*g_rad))
    eps   = np.radians(23.439 - 0.0000004 * d)
    ra_sun = float(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam)) % (2*np.pi))

    # LTAN = 12 + (RAAN - RA_sun) * 12/π
    ltan_computed = (12.0 + (r['raan'] - ra_sun) * (12.0 / np.pi)) % 24.0
    ltan_expected = 10.5   # "10:30"
    assert abs(ltan_computed - ltan_expected) < 1e-6


# ---------------------------------------------------------------------------
# Descending node: RAAN offset by 12 h vs ascending
# ---------------------------------------------------------------------------

def test_sso_orbit_descending_vs_ascending():
    """Descending-node orbit at LTDN X should equal ascending-node orbit at LTAN X+12."""
    r_asc  = sun_synchronous_orbit(_SSO_ALT, "10:30", node_type='ascending',  epoch=_SSO_EPOCH)
    r_desc = sun_synchronous_orbit(_SSO_ALT, "22:30", node_type='descending', epoch=_SSO_EPOCH)
    # Both specify the same ascending-node local time (10:30 AN == 22:30 DN)
    assert r_asc['raan'] == pytest.approx(r_desc['raan'], abs=1e-9)


# ---------------------------------------------------------------------------
# HH:MM:SS format is parsed correctly
# ---------------------------------------------------------------------------

def test_sso_orbit_seconds_format():
    """'HH:MM:SS' string must produce the same result as the equivalent 'HH:MM'
    with fractional minutes."""
    # "10:30:00" should equal "10:30"
    r_hm  = sun_synchronous_orbit(_SSO_ALT, "10:30",    epoch=_SSO_EPOCH)
    r_hms = sun_synchronous_orbit(_SSO_ALT, "10:30:00", epoch=_SSO_EPOCH)
    assert r_hm['raan'] == pytest.approx(r_hms['raan'], abs=1e-12)

    # "10:30:36" = 10 + 30/60 + 36/3600 = 10.51 h
    r_sec = sun_synchronous_orbit(_SSO_ALT, "10:30:36", epoch=_SSO_EPOCH)
    assert r_sec['raan'] != pytest.approx(r_hm['raan'], abs=1e-9)


# ---------------------------------------------------------------------------
# Epoch is stored as datetime64[us]
# ---------------------------------------------------------------------------

def test_sso_orbit_epoch_type():
    """epoch in the returned dict must be datetime64[us]."""
    r = _sso()
    assert r['epoch'].dtype == np.dtype('datetime64[us]')
    assert r['epoch'] == _SSO_EPOCH


# ---------------------------------------------------------------------------
# Result passes straight into propagate_analytical without error
# ---------------------------------------------------------------------------

def test_sso_orbit_propagatable():
    """Unpacking the result dict into propagate_analytical must not raise."""
    params = _sso()
    t = np.array([_SSO_EPOCH, _SSO_EPOCH + np.timedelta64(90 * 60, 's')])
    r, v = propagate_analytical(t, **params)
    assert r.shape == (2, 3)
    assert v.shape == (2, 3)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,match", [
    ({"altitude": _SSO_ALT, "local_time_at_node": "bad"},         "HH:MM"),
    ({"altitude": _SSO_ALT, "local_time_at_node": "10:30",
      "node_type": "sideways"},                                     "node_type"),
    ({"altitude": -1.0,     "local_time_at_node": "10:30"},        "altitude"),
])
def test_sso_orbit_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sun_synchronous_orbit(**kwargs)


# ===========================================================================
# geostationary_orbit
# ===========================================================================

_GEO_EPOCH = np.datetime64('2025-01-01T12:00:00', 'us')
_GEO_LON   = 45.0   # degrees East

_REQUIRED_KEYS = {
    'epoch', 'a', 'e', 'i', 'arg_p', 'raan', 'ma',
    'central_body_mu', 'central_body_j2', 'central_body_radius',
}


def _geo():
    return geostationary_orbit(longitude_deg=_GEO_LON, epoch=_GEO_EPOCH)


class TestGeostationary:

    def test_returns_required_keys(self):
        assert set(_geo().keys()) == _REQUIRED_KEYS

    def test_semi_major_axis(self):
        """GEO semi-major axis should be ~42 164 km."""
        np.testing.assert_allclose(_geo()['a'] / 1000.0, 42164.17, atol=1.0)

    def test_circular_equatorial(self):
        r = _geo()
        assert r['e'] == 0.0
        assert r['i'] == 0.0

    def test_satellite_at_longitude_at_epoch(self):
        """Propagating at epoch → ECEF longitude must equal the input."""
        r = _geo()
        t = np.array([_GEO_EPOCH])
        r_eci, _ = propagate_analytical(t, **r, type='twobody')
        r_ecef   = eci_to_ecef(r_eci, t)
        lon_deg  = np.degrees(np.arctan2(r_ecef[0, 1], r_ecef[0, 0]))
        np.testing.assert_allclose(lon_deg, _GEO_LON, atol=0.01)

    def test_period_is_sidereal_day(self):
        """Orbital period from semi-major axis must equal the sidereal day."""
        r = _geo()
        T = 2.0 * np.pi * np.sqrt(r['a'] ** 3 / EARTH_MU)
        np.testing.assert_allclose(T, 86164.1, atol=1.0)

    def test_default_epoch(self):
        r = geostationary_orbit(longitude_deg=0.0)
        assert r['epoch'] == np.datetime64('2000-01-01T12:00:00', 'us')

    def test_propagates_without_error(self):
        r = _geo()
        t = np.array([_GEO_EPOCH, _GEO_EPOCH + np.timedelta64(3600, 's')])
        pos, vel = propagate_analytical(t, **r, type='twobody')
        assert pos.shape == (2, 3)
        assert vel.shape == (2, 3)


# ===========================================================================
# highly_elliptical_orbit
# ===========================================================================

_HEO_EPOCH  = np.datetime64('2025-06-21T00:00:00', 'us')
_HEO_PERIOD = 43200.0   # 12-hour Molniya-style orbit (s)
_HEO_E      = 0.74
_HEO_LON    = 40.0      # degrees East
_HEO_TIME   = '14:00'   # local mean solar time at apogee


def _heo(arg_p_deg=270.0):
    return highly_elliptical_orbit(
        period_s             = _HEO_PERIOD,
        e                    = _HEO_E,
        epoch                = _HEO_EPOCH,
        apogee_solar_time    = _HEO_TIME,
        apogee_longitude_deg = _HEO_LON,
        arg_p_deg            = arg_p_deg,
    )


def _first_apogee(params, period_s):
    """Return the time of the first apogee after epoch."""
    n         = 2.0 * np.pi / period_s
    delta_t_s = ((np.pi - params['ma']) % (2.0 * np.pi)) / n
    return params['epoch'] + np.timedelta64(int(delta_t_s * 1e6), 'us')


class TestHEO:

    def test_returns_required_keys(self):
        assert set(_heo().keys()) == _REQUIRED_KEYS

    def test_semi_major_axis_from_period(self):
        a_expected = (EARTH_MU * (_HEO_PERIOD / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
        np.testing.assert_allclose(_heo()['a'], a_expected, rtol=1e-10)

    def test_critical_inclination_northern(self):
        """arg_p ≈ 270° → i ≈ 63.435° (northern hemisphere apogee)."""
        r = _heo(arg_p_deg=270.0)
        np.testing.assert_allclose(np.degrees(r['i']), 63.4349, atol=0.001)

    def test_critical_inclination_southern(self):
        """arg_p ≈ 90° → i ≈ 116.565° (southern hemisphere apogee)."""
        r = _heo(arg_p_deg=90.0)
        np.testing.assert_allclose(np.degrees(r['i']), 116.5651, atol=0.001)

    def test_apogee_longitude(self):
        """At the first apogee, ECEF longitude must match the requested value."""
        r      = _heo()
        T_apo  = _first_apogee(r, _HEO_PERIOD)
        r_eci, _ = propagate_analytical(np.array([T_apo]), **r, type='twobody')
        r_ecef   = eci_to_ecef(r_eci, np.array([T_apo]))
        lon_deg  = np.degrees(np.arctan2(r_ecef[0, 1], r_ecef[0, 0]))
        np.testing.assert_allclose(lon_deg, _HEO_LON, atol=0.01)

    def test_apogee_solar_time(self):
        """At the first apogee, local mean solar time must match the request."""
        r     = _heo()
        T_apo = _first_apogee(r, _HEO_PERIOD)
        # UTC hours at apogee
        T_day = T_apo.astype('datetime64[D]').astype('datetime64[us]')
        utc_h = float((T_apo - T_day).astype(np.float64) * 1e-6 / 3600.0)
        # Local mean solar time = UTC + longitude / 15
        lmst_h   = (utc_h + _HEO_LON / 15.0) % 24.0
        expected = 14.0   # '14:00' = 14.0 decimal hours
        assert min(abs(lmst_h - expected), 24.0 - abs(lmst_h - expected)) < 5.0 / 60.0

    def test_propagates_without_error(self):
        r = _heo()
        t = np.array([_HEO_EPOCH, _HEO_EPOCH + np.timedelta64(3600, 's')])
        pos, vel = propagate_analytical(t, **r, type='twobody')
        assert pos.shape == (2, 3)

    def test_invalid_eccentricity_zero(self):
        with pytest.raises(ValueError, match="0 < e < 1"):
            highly_elliptical_orbit(
                period_s=43200, e=0.0, epoch=_HEO_EPOCH,
                apogee_solar_time='12:00', apogee_longitude_deg=0.0,
            )

    def test_invalid_eccentricity_one(self):
        with pytest.raises(ValueError, match="0 < e < 1"):
            highly_elliptical_orbit(
                period_s=43200, e=1.0, epoch=_HEO_EPOCH,
                apogee_solar_time='12:00', apogee_longitude_deg=0.0,
            )

    def test_invalid_period(self):
        with pytest.raises(ValueError, match="period_s"):
            highly_elliptical_orbit(
                period_s=-1, e=0.74, epoch=_HEO_EPOCH,
                apogee_solar_time='12:00', apogee_longitude_deg=0.0,
            )
