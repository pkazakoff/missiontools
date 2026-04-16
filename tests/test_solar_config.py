"""Tests for missiontools.power.solar_config — solar power configuration."""

import numpy as np
import pytest

from missiontools import Spacecraft, FixedAttitudeLaw, NormalVectorSolarConfig
from missiontools.power import AbstractSolarConfig
from missiontools.orbit import sun_vec_eci

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EPOCH = np.datetime64('2025-06-21T12:00:00', 'us')

_KW = dict(
    a      = 6_771_000.0,
    e      = 0.0,
    i      = np.radians(51.6),
    raan   = 0.0,
    arg_p  = 0.0,
    ma     = 0.0,
    epoch  = _EPOCH,
)


def _sc(**overrides):
    kw = {**_KW, **overrides}
    return Spacecraft(**kw)


def _simple_config(efficiency=0.3):
    """Single zenith-facing panel (body-z = boresight = nadir for nadir pointing)."""
    return NormalVectorSolarConfig(
        normal_vecs=[[0, 0, 1]],
        areas=[1.0],
        efficiency=efficiency,
    )


# ===========================================================================
# AbstractSolarConfig validation
# ===========================================================================

class TestAbstractSolarConfig:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractSolarConfig(0.5)

    def test_efficiency_too_low(self):
        with pytest.raises(ValueError, match='efficiency'):
            NormalVectorSolarConfig([[0, 0, 1]], [1.0], efficiency=0.0)

    def test_efficiency_too_high(self):
        with pytest.raises(ValueError, match='efficiency'):
            NormalVectorSolarConfig([[0, 0, 1]], [1.0], efficiency=1.5)

    def test_negative_efficiency(self):
        with pytest.raises(ValueError, match='efficiency'):
            NormalVectorSolarConfig([[0, 0, 1]], [1.0], efficiency=-0.1)


# ===========================================================================
# NormalVectorSolarConfig construction
# ===========================================================================

class TestNormalVectorConstruct:

    def test_basic_construction(self):
        cfg = NormalVectorSolarConfig(
            normal_vecs=[[1, 0, 0], [0, 1, 0]],
            areas=[2.0, 3.0],
            efficiency=0.28,
        )
        assert cfg.efficiency == 0.28
        assert cfg.normals.shape == (2, 3)
        assert cfg.areas.shape == (2,)

    def test_normals_are_normalised(self):
        cfg = NormalVectorSolarConfig(
            normal_vecs=[[3, 0, 0]],
            areas=[1.0],
            efficiency=0.3,
        )
        np.testing.assert_allclose(np.linalg.norm(cfg.normals, axis=1), 1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            NormalVectorSolarConfig([[0, 0, 1], [1, 0, 0]], [1.0], efficiency=0.3)

    def test_bad_normal_shape_raises(self):
        with pytest.raises(ValueError, match='normal_vecs'):
            NormalVectorSolarConfig([[0, 0]], [1.0], efficiency=0.3)

    def test_zero_area_raises(self):
        with pytest.raises(ValueError, match='positive'):
            NormalVectorSolarConfig([[0, 0, 1]], [0.0], efficiency=0.3)

    def test_zero_normal_raises(self):
        with pytest.raises(ValueError, match='non-zero'):
            NormalVectorSolarConfig([[0, 0, 0]], [1.0], efficiency=0.3)

    def test_read_only_copies(self):
        cfg = NormalVectorSolarConfig([[0, 0, 1]], [1.0], efficiency=0.3)
        n = cfg.normals
        n[0, 0] = 999.0
        assert cfg.normals[0, 0] != 999.0  # internal state unchanged


# ===========================================================================
# Spacecraft attachment
# ===========================================================================

class TestAttachment:

    def test_add_solar_config(self):
        sc = _sc()
        cfg = _simple_config()
        sc.add_solar_config(cfg)
        assert cfg.spacecraft is sc
        assert len(sc.solar_configs) == 1
        assert sc.solar_configs[0] is cfg

    def test_multiple_configs(self):
        sc = _sc()
        c1 = _simple_config()
        c2 = _simple_config()
        sc.add_solar_config(c1)
        sc.add_solar_config(c2)
        assert len(sc.solar_configs) == 2

    def test_add_wrong_type_raises(self):
        sc = _sc()
        with pytest.raises(TypeError, match='AbstractSolarConfig'):
            sc.add_solar_config("not a config")

    def test_solar_configs_read_only(self):
        sc = _sc()
        sc.add_solar_config(_simple_config())
        configs = sc.solar_configs
        configs.clear()
        assert len(sc.solar_configs) == 1  # internal list unchanged

    def test_unattached_generation_raises(self):
        cfg = _simple_config()
        with pytest.raises(RuntimeError, match='attached'):
            cfg.generation(_EPOCH, _EPOCH + np.timedelta64(60, 's'),
                           np.timedelta64(60, 's'))

    def test_unattached_oap_raises(self):
        cfg = _simple_config()
        with pytest.raises(RuntimeError, match='attached'):
            cfg.oap()


# ===========================================================================
# generation()
# ===========================================================================

class TestGeneration:

    def test_returns_dict_with_correct_keys(self):
        sc = _sc()
        cfg = _simple_config()
        sc.add_solar_config(cfg)
        t_end = _EPOCH + np.timedelta64(600, 's')
        result = cfg.generation(_EPOCH, t_end, np.timedelta64(60, 's'))
        assert 't' in result
        assert 'power' in result
        assert len(result['t']) == len(result['power'])

    def test_power_non_negative(self):
        sc = _sc()
        cfg = _simple_config()
        sc.add_solar_config(cfg)
        t_end = _EPOCH + np.timedelta64(3600, 's')
        result = cfg.generation(_EPOCH, t_end, np.timedelta64(60, 's'))
        assert np.all(result['power'] >= 0.0)

    def test_eclipsed_power_is_zero(self):
        """Spacecraft starting directly behind Earth should have zero power."""
        sun = sun_vec_eci(_EPOCH)
        # Place spacecraft on anti-sun side.  ma=pi puts it at the opposite
        # side from epoch; we hack the RAAN to align the orbit anti-sunward.
        # Simpler: just check that zero power exists at some point in an orbit.
        sc = _sc()
        cfg = _simple_config()
        sc.add_solar_config(cfg)
        period_s = 2 * np.pi * np.sqrt(sc.a**3 / sc.central_body_mu)
        period = np.timedelta64(int(period_s * 1e6), 'us')
        result = cfg.generation(_EPOCH, _EPOCH + period, np.timedelta64(30, 's'))
        # For a LEO orbit, there must be some eclipse time
        assert np.any(result['power'] == 0.0)

    def test_efficiency_scales_power(self):
        """Doubling efficiency should double the power."""
        sc1 = _sc()
        cfg1 = NormalVectorSolarConfig([[0, 0, 1]], [1.0], efficiency=0.15)
        sc1.add_solar_config(cfg1)

        sc2 = _sc()
        cfg2 = NormalVectorSolarConfig([[0, 0, 1]], [1.0], efficiency=0.30)
        sc2.add_solar_config(cfg2)

        t_end = _EPOCH + np.timedelta64(300, 's')
        step = np.timedelta64(60, 's')
        p1 = cfg1.generation(_EPOCH, t_end, step)['power']
        p2 = cfg2.generation(_EPOCH, t_end, step)['power']

        # Where power is non-zero, ratio should be 2.0
        mask = p1 > 0
        if np.any(mask):
            np.testing.assert_allclose(p2[mask] / p1[mask], 2.0, rtol=1e-10)

    def test_empty_time_range(self):
        sc = _sc()
        cfg = _simple_config()
        sc.add_solar_config(cfg)
        result = cfg.generation(
            _EPOCH, _EPOCH - np.timedelta64(1, 's'), np.timedelta64(60, 's'),
        )
        assert len(result['t']) == 0
        assert len(result['power']) == 0

    def test_irradiance_scales_power(self):
        sc = _sc()
        cfg = _simple_config()
        sc.add_solar_config(cfg)
        t_end = _EPOCH + np.timedelta64(300, 's')
        step = np.timedelta64(60, 's')
        p1 = cfg.generation(_EPOCH, t_end, step, irradiance=1000.0)['power']
        p2 = cfg.generation(_EPOCH, t_end, step, irradiance=2000.0)['power']
        mask = p1 > 0
        if np.any(mask):
            np.testing.assert_allclose(p2[mask] / p1[mask], 2.0, rtol=1e-10)


# ===========================================================================
# optimal_angle()
# ===========================================================================

class TestOptimalAngle:

    def test_single_panel_z_axis_rotation(self):
        """Single panel with normal [1,0,0], rotating about z-axis.

        The optimal angle should align the candidate sun direction with -x
        (so the panel normal [1,0,0] faces the 'sun').
        """
        cfg = NormalVectorSolarConfig(
            normal_vecs=[[1, 0, 0]],
            areas=[1.0],
            efficiency=0.3,
        )
        theta = cfg.optimal_angle([0, 0, 1])
        # The reference basis is deterministic — just verify the projected area
        # is maximised at the returned angle.
        axis = np.array([0, 0, 1.0])
        # Build the same basis as the implementation
        cardinals = np.eye(3)
        dots = np.abs(cardinals @ axis)
        least = cardinals[np.argmin(dots)]
        u = np.cross(axis, least)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)

        d = np.cos(theta) * u + np.sin(theta) * v
        proj_at_opt = max(float(np.dot(-d, [1, 0, 0])), 0.0)
        assert proj_at_opt > 0.99  # near-maximum (1.0 for unit area)

    def test_symmetric_panels_any_angle_ok(self):
        """Two opposing panels — all angles give the same projected area."""
        cfg = NormalVectorSolarConfig(
            normal_vecs=[[1, 0, 0], [-1, 0, 0]],
            areas=[1.0, 1.0],
            efficiency=0.3,
        )
        theta = cfg.optimal_angle([0, 0, 1])
        # All angles should yield projected area ≈ 1.0 (one panel always faces 'sun')
        axis = np.array([0, 0, 1.0])
        cardinals = np.eye(3)
        dots = np.abs(cardinals @ axis)
        least = cardinals[np.argmin(dots)]
        u = np.cross(axis, least)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        d = np.cos(theta) * u + np.sin(theta) * v
        total = sum(
            a * max(float(np.dot(-d, n)), 0.0)
            for n, a in zip(cfg.normals, cfg.areas)
        )
        assert total == pytest.approx(1.0, abs=0.02)

    def test_returns_float(self):
        cfg = _simple_config()
        theta = cfg.optimal_angle([0, 0, 1])
        assert isinstance(theta, float)
        assert 0 <= theta < 2 * np.pi


# ===========================================================================
# oap()
# ===========================================================================

class TestOAP:

    def test_returns_scalar(self):
        sc = _sc()
        cfg = _simple_config()
        sc.add_solar_config(cfg)
        power = cfg.oap()
        assert isinstance(power, float)

    def test_oap_non_negative(self):
        sc = _sc()
        cfg = _simple_config()
        sc.add_solar_config(cfg)
        assert cfg.oap() >= 0.0

    def test_oap_less_than_max(self):
        """OAP must be less than irradiance * area * efficiency (due to eclipse + geometry)."""
        sc = _sc()
        area = 5.0
        eff = 0.3
        cfg = NormalVectorSolarConfig([[0, 0, 1]], [area], efficiency=eff)
        sc.add_solar_config(cfg)
        max_possible = 1366.0 * area * eff
        assert cfg.oap() < max_possible

    def test_custom_start_time(self):
        sc = _sc()
        cfg = _simple_config()
        sc.add_solar_config(cfg)
        t_custom = _EPOCH + np.timedelta64(3600, 's')
        power = cfg.oap(start_time=t_custom)
        assert isinstance(power, float)
        assert power >= 0.0
