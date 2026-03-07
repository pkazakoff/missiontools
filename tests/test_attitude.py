import numpy as np
import pytest

from missiontools import AttitudeLaw, Spacecraft
from missiontools.orbit.propagation import propagate_analytical
from missiontools.attitude.attitude_law import _q_boresight

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_EPOCH = np.datetime64('2025-01-01T00:00:00', 'us')
_T2    = np.array([_EPOCH, _EPOCH + np.timedelta64(5400, 's')])  # 2 timesteps

_SC_KW = dict(
    a      = 6_771_000.0,
    e      = 0.0,
    i      = np.radians(51.6),
    raan   = 0.0,
    arg_p  = 0.0,
    ma     = 0.0,
    epoch  = _EPOCH,
)


def _orbit_state(t=_T2):
    """Return (r, v, t) for a circular test orbit."""
    r, v = propagate_analytical(t, **_SC_KW, propagator_type='twobody')
    return r, v, t


# ===========================================================================
# Fixed attitude laws
# ===========================================================================

class TestAttitudeLawFixed:

    def test_nadir_boresight_in_lvlh(self):
        """Stored quaternion must encode the nadir direction [-1,0,0] in LVLH."""
        law = AttitudeLaw.nadir()
        assert law._mode  == 'fixed'
        assert law._frame == 'lvlh'
        np.testing.assert_allclose(_q_boresight(law._q),
                                   [-1.0, 0.0, 0.0], atol=1e-12)

    def test_nadir_pointing_lvlh_is_minus_r_hat(self):
        """Nadir pointing in LVLH must be [-1,0,0] at every timestep."""
        law = AttitudeLaw.nadir()
        r, v, t = _orbit_state()
        p = law.pointing_lvlh(r, v, t)
        np.testing.assert_allclose(p, np.tile([-1., 0., 0.], (len(t), 1)),
                                   atol=1e-10)

    def test_fixed_eci_pointing_eci_constant(self):
        """Fixed ECI [0,1,0] must return [0,1,0] at every timestep."""
        law = AttitudeLaw.fixed([0., 1., 0.], 'eci')
        r, v, t = _orbit_state()
        p = law.pointing_eci(r, v, t)
        np.testing.assert_allclose(p, np.tile([0., 1., 0.], (len(t), 1)),
                                   atol=1e-12)

    def test_fixed_ecef_pointing_ecef_constant(self):
        """Fixed ECEF [0,0,1] must return [0,0,1] from pointing_ecef."""
        law = AttitudeLaw.fixed([0., 0., 1.], 'ecef')
        r, v, t = _orbit_state()
        p = law.pointing_ecef(r, v, t)
        np.testing.assert_allclose(p, np.tile([0., 0., 1.], (len(t), 1)),
                                   atol=1e-12)

    def test_pointing_eci_unit_norm(self):
        law = AttitudeLaw.nadir()
        r, v, t = _orbit_state()
        norms = np.linalg.norm(law.pointing_eci(r, v, t), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_pointing_lvlh_unit_norm(self):
        law = AttitudeLaw.fixed([1., 1., 0.], 'eci')
        r, v, t = _orbit_state()
        norms = np.linalg.norm(law.pointing_lvlh(r, v, t), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_pointing_ecef_unit_norm(self):
        law = AttitudeLaw.nadir()
        r, v, t = _orbit_state()
        norms = np.linalg.norm(law.pointing_ecef(r, v, t), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_scalar_input_returns_1d(self):
        """Single timestep (1-D r/v, scalar t) must return 1-D (3,) arrays."""
        law = AttitudeLaw.nadir()
        r, v, t = _orbit_state(t=np.array([_EPOCH]))
        p_eci  = law.pointing_eci(r[0],  v[0],  t[0])
        p_lvlh = law.pointing_lvlh(r[0], v[0],  t[0])
        p_ecef = law.pointing_ecef(r[0], v[0],  t[0])
        assert p_eci.shape  == (3,)
        assert p_lvlh.shape == (3,)
        assert p_ecef.shape == (3,)

    def test_roll_changes_orientation_not_boresight(self):
        """Adding a roll must not change the boresight direction."""
        vec  = [0., 0., 1.]
        law0 = AttitudeLaw.fixed(vec, 'eci', roll=0.0)
        law1 = AttitudeLaw.fixed(vec, 'eci', roll=np.pi / 4)
        # Quaternions differ
        assert not np.allclose(law0._q, law1._q)
        # But the boresight (pointing direction) is the same
        np.testing.assert_allclose(law0._pointing_in_ref,
                                   law1._pointing_in_ref, atol=1e-12)

    def test_invalid_frame_raises(self):
        with pytest.raises(ValueError, match="frame"):
            AttitudeLaw.fixed([1., 0., 0.], 'xyz')

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero"):
            AttitudeLaw.fixed([0., 0., 0.], 'eci')

    def test_unnormalized_vector_accepted(self):
        """Vectors of arbitrary magnitude should be normalised silently."""
        law = AttitudeLaw.fixed([0., 0., 5.], 'eci')
        np.testing.assert_allclose(law._pointing_in_ref, [0., 0., 1.],
                                   atol=1e-12)

    def test_fixed_lvlh_changes_in_eci(self):
        """A fixed LVLH vector should produce *different* ECI directions at
        different timesteps (because LVLH rotates with the orbit)."""
        law  = AttitudeLaw.fixed([0., 1., 0.], 'lvlh')   # along-track
        r, v, t = _orbit_state()
        p = law.pointing_eci(r, v, t)
        # The two ECI directions should differ (orbit has moved)
        assert not np.allclose(p[0], p[1], atol=1e-4)


class TestNadirRoll:

    def test_roll_zero_same_as_default(self):
        law0 = AttitudeLaw.nadir()
        law1 = AttitudeLaw.nadir(roll=0.0)
        np.testing.assert_allclose(law0._q, law1._q, atol=1e-15)

    def test_boresight_unchanged_with_roll(self):
        """Roll rotates about boresight, so the boresight direction must stay [-1,0,0]."""
        for roll in [0.3, -0.5, np.pi]:
            law = AttitudeLaw.nadir(roll=roll)
            np.testing.assert_allclose(_q_boresight(law._q),
                                       [-1.0, 0.0, 0.0], atol=1e-12)

    def test_roll_rotates_body_x(self):
        """Body-x should rotate in the LVLH S-W plane by the roll angle."""
        from missiontools.attitude.attitude_law import _q_rotate
        roll = np.radians(45)
        law = AttitudeLaw.nadir(roll=roll)
        body_x_in_lvlh = _q_rotate(law._q, np.array([1., 0., 0.]))
        # At roll=0, body-x = S-hat = [0,1,0] in LVLH
        # Roll rotates in the plane perpendicular to boresight (nadir)
        expected = np.array([0., np.cos(roll), -np.sin(roll)])
        np.testing.assert_allclose(body_x_in_lvlh, expected, atol=1e-12)

    def test_full_rotation_returns_to_start(self):
        """2*pi roll gives the same physical rotation (q and -q are equivalent)."""
        law0 = AttitudeLaw.nadir()
        law_full = AttitudeLaw.nadir(roll=2 * np.pi)
        # Quaternion double cover: q and -q represent the same rotation
        sign = np.sign(law0._q[0] * law_full._q[0])
        np.testing.assert_allclose(law0._q, sign * law_full._q, atol=1e-12)


# ===========================================================================
# Target-tracking attitude laws
# ===========================================================================

class TestAttitudeLawTrack:

    def _make_target(self):
        """A second spacecraft offset in RAAN so positions differ."""
        return Spacecraft(
            a      = 6_771_000.0,
            e      = 0.0,
            i      = np.radians(51.6),
            raan   = np.radians(90.0),   # 90° offset → different position
            arg_p  = 0.0,
            ma     = 0.0,
            epoch  = _EPOCH,
        )

    def test_track_pointing_eci_unit_norm(self):
        target = self._make_target()
        law = AttitudeLaw.track(target)
        r, v, t = _orbit_state()
        norms = np.linalg.norm(law.pointing_eci(r, v, t), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_track_points_toward_target(self):
        target = self._make_target()
        law = AttitudeLaw.track(target)
        r, v, t = _orbit_state()
        p_eci = law.pointing_eci(r, v, t)

        # Manually compute expected direction
        r_tgt, _ = propagate_analytical(t, **target.keplerian_params,
                                        propagator_type='twobody')
        diff = r_tgt - r
        expected = diff / np.linalg.norm(diff, axis=1, keepdims=True)
        np.testing.assert_allclose(p_eci, expected, atol=1e-12)

    def test_track_invalid_target_raises(self):
        with pytest.raises(TypeError, match="Spacecraft"):
            AttitudeLaw.track("not a spacecraft")


# ===========================================================================
# Yaw steering
# ===========================================================================

class TestYawSteering:

    def _make_solar_config(self):
        """Single +x-facing panel — optimal angle should steer +x toward sun."""
        from missiontools import NormalVectorSolarConfig
        return NormalVectorSolarConfig(
            normal_vecs=[[1, 0, 0]],
            areas=[1.0],
            efficiency=0.3,
        )

    def test_activate_and_deactivate(self):
        law = AttitudeLaw.nadir()
        cfg = self._make_solar_config()
        law.yaw_steering(cfg)
        assert law._solar_config is cfg
        assert law._yaw_opt_dir is not None

        law.yaw_steering(None)
        assert law._solar_config is None
        assert law._yaw_opt_dir is None

    def test_invalid_type_raises(self):
        law = AttitudeLaw.nadir()
        with pytest.raises(TypeError, match='AbstractSolarConfig'):
            law.yaw_steering("not a config")

    def test_boresight_unchanged(self):
        """Yaw steering must not alter the boresight direction."""
        law = AttitudeLaw.nadir()
        cfg = self._make_solar_config()
        r, v, t = _orbit_state()

        p_before = law.pointing_eci(r, v, t).copy()
        law.yaw_steering(cfg)
        p_after = law.pointing_eci(r, v, t)
        np.testing.assert_allclose(p_after, p_before, atol=1e-12)

    def test_rotate_from_body_changes(self):
        """With yaw steering, rotate_from_body should differ from static roll."""
        law = AttitudeLaw.nadir()
        cfg = self._make_solar_config()
        r, v, t = _orbit_state()

        static = law.rotate_from_body([1, 0, 0], r, v, t).copy()
        law.yaw_steering(cfg)
        steered = law.rotate_from_body([1, 0, 0], r, v, t)
        # The steered result should generally differ from static
        assert not np.allclose(static, steered, atol=1e-6)

    def test_optimal_direction_faces_sun(self):
        """The optimal body-frame direction should be closer to the sun
        with yaw steering than without."""
        from missiontools.orbit.frames import sun_vec_eci
        law = AttitudeLaw.nadir()
        cfg = self._make_solar_config()
        r, v, t = _orbit_state()

        # Get the optimal body-frame direction
        theta = cfg.optimal_angle(np.array([0., 0., 1.]))
        d_opt = np.array([np.sin(theta), -np.cos(theta), 0.0])

        # Without yaw steering
        law_static = AttitudeLaw.nadir()
        d_eci_static = np.atleast_2d(law_static.rotate_from_body(d_opt, r, v, t))

        # With yaw steering
        law.yaw_steering(cfg)
        d_eci_steered = np.atleast_2d(law.rotate_from_body(d_opt, r, v, t))

        # Sun direction
        sun = np.atleast_2d(sun_vec_eci(t))

        # Steered alignment should be >= static alignment
        cos_static  = np.einsum('ij,ij->i', d_eci_static, sun)
        cos_steered = np.einsum('ij,ij->i', d_eci_steered, sun)
        assert np.all(cos_steered >= cos_static - 1e-10)

    def test_yaw_steering_with_track_mode(self):
        """Yaw steering should also work in track mode."""
        target = Spacecraft(**{**_SC_KW, 'raan': np.radians(90.0)})
        law = AttitudeLaw.track(target)
        cfg = self._make_solar_config()
        r, v, t = _orbit_state()

        static = law.rotate_from_body([1, 0, 0], r, v, t).copy()
        law.yaw_steering(cfg)
        steered = law.rotate_from_body([1, 0, 0], r, v, t)
        # Should differ
        assert not np.allclose(static, steered, atol=1e-6)
