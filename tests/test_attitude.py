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
# Limb-pointing attitude laws
# ===========================================================================

class TestAttitudeLawLimb:

    # -- validation --------------------------------------------------------

    def test_negative_altitude_raises(self):
        with pytest.raises(ValueError, match="altitude_km"):
            AttitudeLaw.limb([0., 0., 1.], altitude_km=-1.0)

    def test_zero_body_vector_raises(self):
        with pytest.raises(ValueError, match="zero"):
            AttitudeLaw.limb([0., 0., 0.], altitude_km=0.0)

    def test_bad_body_vector_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            AttitudeLaw.limb([0., 0., 1., 0.], altitude_km=0.0)

    def test_bad_flattening_raises(self):
        with pytest.raises(ValueError, match="flattening"):
            AttitudeLaw.limb([0., 0., 1.], altitude_km=0.0,
                             body_flattening=1.5)
        with pytest.raises(ValueError, match="flattening"):
            AttitudeLaw.limb([0., 0., 1.], altitude_km=0.0,
                             body_flattening=-0.1)

    def test_bad_semi_major_raises(self):
        with pytest.raises(ValueError, match="body_semi_major_axis"):
            AttitudeLaw.limb([0., 0., 1.], altitude_km=0.0,
                             body_semi_major_axis=-1.0)

    def test_spacecraft_inside_ellipsoid_raises(self):
        """Altitude so large that SC is inside the offset ellipsoid → raise."""
        from missiontools.orbit.constants import EARTH_SEMI_MAJOR_AXIS
        # Orbit radius 6_771_000 m; altitude of 500 km → offset a = 6.878e6
        # — SC sits inside, tangent geometry undefined.
        law = AttitudeLaw.limb([0., 0., 1.], altitude_km=500.0,
                               body_flattening=0.0)
        r, v, t = _orbit_state()
        with pytest.raises(ValueError, match="inside the offset ellipsoid"):
            law.pointing_eci(r, v, t)

    # -- geometry ---------------------------------------------------------

    def test_spherical_off_nadir_analytic(self):
        """Spherical body (flattening=0), altitude=0, yaw=0, body-z limb:
        pointing_lvlh must equal (−cos(off), sin(off), 0) with
        off = asin(R_body / r_sc)."""
        from missiontools.orbit.constants import EARTH_SEMI_MAJOR_AXIS
        law = AttitudeLaw.limb([0., 0., 1.], altitude_km=0.0,
                               body_flattening=0.0)
        r, v, t = _orbit_state()
        p_lvlh = law.pointing_lvlh(r, v, t)

        r_sc = np.linalg.norm(r, axis=1)                     # (N,)
        off  = np.arcsin(EARTH_SEMI_MAJOR_AXIS / r_sc)
        expected = np.stack([-np.cos(off), np.sin(off),
                             np.zeros_like(off)], axis=1)
        np.testing.assert_allclose(p_lvlh, expected, atol=1e-10)

    def test_tangent_point_on_offset_ellipsoid(self):
        """The ray should be tangent to the offset ellipsoid: the quadratic
        distance-along-ray has a double root and the touch point satisfies
        (x/A)² + (y/A)² + (z/B)² = 1 in ECEF."""
        from missiontools.attitude.attitude_law import _q_boresight  # noqa
        from missiontools.orbit.constants import (EARTH_SEMI_MAJOR_AXIS,
                                                  EARTH_INVERSE_FLATTENING)
        from missiontools.orbit.frames import eci_to_ecef

        altitude_km = 20.0
        law = AttitudeLaw.limb([0., 0., 1.], altitude_km=altitude_km,
                               yaw_deg=30.0)
        r, v, t = _orbit_state()
        d_eci = law.pointing_eci(r, v, t)

        # Convert SC position and pointing direction to ECEF
        r_ecef = eci_to_ecef(r, t)
        d_ecef = eci_to_ecef(d_eci, t)

        f = 1.0 / EARTH_INVERSE_FLATTENING
        A = EARTH_SEMI_MAJOR_AXIS + altitude_km * 1e3
        B = EARTH_SEMI_MAJOR_AXIS * (1 - f) + altitude_km * 1e3
        diag_Q = np.array([1/A**2, 1/A**2, 1/B**2])

        # Distance-along-ray quadratic: α s² + 2β s + γ = 0
        alpha = (d_ecef**2 * diag_Q).sum(axis=1)
        beta  = (d_ecef * r_ecef * diag_Q).sum(axis=1)
        gamma = (r_ecef**2 * diag_Q).sum(axis=1) - 1.0

        disc = beta**2 - alpha * gamma
        np.testing.assert_allclose(disc, 0.0, atol=1e-6)

        # Touch point (double root): s = −β/α, verify it's on the ellipsoid
        s = -beta / alpha
        touch = r_ecef + s[:, None] * d_ecef
        np.testing.assert_allclose((touch**2 * diag_Q).sum(axis=1), 1.0,
                                   atol=1e-9)

    def test_ellipsoid_differs_from_sphere_at_high_inclination(self):
        """With a 51.6° orbit, the WGS84 ellipsoid off-nadir angle must
        differ from the spherical answer by a detectable amount."""
        law_ell = AttitudeLaw.limb([0., 0., 1.], altitude_km=0.0)
        law_sph = AttitudeLaw.limb([0., 0., 1.], altitude_km=0.0,
                                   body_flattening=0.0)
        r, v, t = _orbit_state()
        p_ell = law_ell.pointing_lvlh(r, v, t)
        p_sph = law_sph.pointing_lvlh(r, v, t)
        diff = np.linalg.norm(p_ell - p_sph, axis=1)
        assert np.any(diff > 1e-4), \
            "WGS84 ellipsoid should measurably differ from sphere"

    def test_yaw_plus_90_selects_plus_W(self):
        """yaw=+90° places the limb direction on the +Ŵ side of LVLH."""
        law = AttitudeLaw.limb([0., 0., 1.], altitude_km=0.0,
                               body_flattening=0.0, yaw_deg=90.0)
        r, v, t = _orbit_state()
        p_lvlh = law.pointing_lvlh(r, v, t)
        # Expected: (−cos(off), 0, +sin(off))
        assert np.all(p_lvlh[:, 1] < 1e-10)    # no along-track component
        assert np.all(p_lvlh[:, 2] > 0)        # +Ŵ component

    def test_yaw_minus_90_selects_minus_W(self):
        law = AttitudeLaw.limb([0., 0., 1.], altitude_km=0.0,
                               body_flattening=0.0, yaw_deg=-90.0)
        r, v, t = _orbit_state()
        p_lvlh = law.pointing_lvlh(r, v, t)
        assert np.all(p_lvlh[:, 1] < 1e-10)
        assert np.all(p_lvlh[:, 2] < 0)

    # -- body_vector freedom ----------------------------------------------

    def test_body_vector_x_aligns_with_limb_direction(self):
        """rotate_from_body([1,0,0], ...) on a body-x limb law must give the
        same ECI direction as the boresight of a body-z limb law (same
        geometry)."""
        law_x = AttitudeLaw.limb([1., 0., 0.], altitude_km=0.0)
        law_z = AttitudeLaw.limb([0., 0., 1.], altitude_km=0.0)
        r, v, t = _orbit_state()
        d_x = law_x.rotate_from_body([1., 0., 0.], r, v, t)
        d_z = law_z.pointing_eci(r, v, t)
        np.testing.assert_allclose(d_x, d_z, atol=1e-12)

    def test_body_vector_boresight_not_aligned_when_body_vector_nonstandard(self):
        """With body_vector=[1,0,0] the boresight (body-z) is NOT the limb
        direction — it's perpendicular to it."""
        law = AttitudeLaw.limb([1., 0., 0.], altitude_km=0.0)
        r, v, t = _orbit_state()
        d_limb = law.rotate_from_body([1., 0., 0.], r, v, t)
        b_eci  = law.pointing_eci(r, v, t)            # body-z in ECI
        dots   = np.einsum('ni,ni->n', d_limb, b_eci)
        np.testing.assert_allclose(dots, 0.0, atol=1e-10)

    # -- roll --------------------------------------------------------------

    def test_roll_invariant_of_aligned_axis(self):
        """Roll about the limb direction must not move the aligned body
        vector."""
        r, v, t = _orbit_state()
        law0 = AttitudeLaw.limb([1., 0., 0.], altitude_km=0.0, roll_deg=0.0)
        law1 = AttitudeLaw.limb([1., 0., 0.], altitude_km=0.0, roll_deg=37.0)
        d0 = law0.rotate_from_body([1., 0., 0.], r, v, t)
        d1 = law1.rotate_from_body([1., 0., 0.], r, v, t)
        np.testing.assert_allclose(d0, d1, atol=1e-12)

    def test_roll_moves_perpendicular_body_axis(self):
        """Roll must move a body axis perpendicular to the limb direction."""
        r, v, t = _orbit_state()
        law0 = AttitudeLaw.limb([1., 0., 0.], altitude_km=0.0, roll_deg=0.0)
        law1 = AttitudeLaw.limb([1., 0., 0.], altitude_km=0.0, roll_deg=45.0)
        y0 = law0.rotate_from_body([0., 1., 0.], r, v, t)
        y1 = law1.rotate_from_body([0., 1., 0.], r, v, t)
        diff = np.linalg.norm(y0 - y1, axis=1)
        assert np.all(diff > 1e-3)

    def test_full_roll_returns_to_start(self):
        """roll=360° gives the same body orientation as roll=0°."""
        r, v, t = _orbit_state()
        law0 = AttitudeLaw.limb([1., 0., 0.], altitude_km=0.0, roll_deg=0.0)
        law_full = AttitudeLaw.limb([1., 0., 0.], altitude_km=0.0,
                                    roll_deg=360.0)
        y0 = law0.rotate_from_body([0., 1., 0.], r, v, t)
        yf = law_full.rotate_from_body([0., 1., 0.], r, v, t)
        np.testing.assert_allclose(y0, yf, atol=1e-10)

    # -- shape / norm ------------------------------------------------------

    def test_scalar_input_returns_1d(self):
        law = AttitudeLaw.limb([0., 0., 1.], altitude_km=10.0)
        r, v, t = _orbit_state(t=np.array([_EPOCH]))
        p_eci  = law.pointing_eci(r[0],  v[0],  t[0])
        p_lvlh = law.pointing_lvlh(r[0], v[0],  t[0])
        p_ecef = law.pointing_ecef(r[0], v[0],  t[0])
        assert p_eci.shape  == (3,)
        assert p_lvlh.shape == (3,)
        assert p_ecef.shape == (3,)

    def test_pointing_unit_norm(self):
        law = AttitudeLaw.limb([0., 0., 1.], altitude_km=30.0, yaw_deg=45.0)
        r, v, t = _orbit_state()
        for p in [law.pointing_eci(r, v, t),
                  law.pointing_lvlh(r, v, t),
                  law.pointing_ecef(r, v, t)]:
            np.testing.assert_allclose(np.linalg.norm(p, axis=1), 1.0,
                                       atol=1e-12)

    # -- interaction with yaw_steering ------------------------------------

    def test_yaw_steering_not_supported(self):
        from missiontools import NormalVectorSolarConfig
        cfg = NormalVectorSolarConfig(
            normal_vecs=[[1, 0, 0]], areas=[1.0], efficiency=0.3,
        )
        law = AttitudeLaw.limb([0., 0., 1.], altitude_km=0.0)
        with pytest.raises(NotImplementedError, match="limb"):
            law.yaw_steering(cfg)


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


# ===========================================================================
# Custom attitude law
# ===========================================================================

def _identity_cb(t, r_eci, v_eci):
    """Callback that returns identity quaternions (body frame = ECI frame)."""
    N = len(t)
    q = np.zeros((N, 4))
    q[:, 0] = 1.0
    return q


def _nadir_equiv_cb(t, r_eci, v_eci):
    """Callback that mimics nadir pointing: body-z = -r_hat in ECI."""
    from missiontools.attitude.attitude_law import _q_from_vec_batch
    nadir = -r_eci / np.linalg.norm(r_eci, axis=1, keepdims=True)
    return _q_from_vec_batch(nadir)


class TestAttitudeLawCustom:

    def test_classmethod_stores_callback(self):
        law = AttitudeLaw.custom(_identity_cb)
        assert law._mode == 'custom'
        assert law._callback is _identity_cb

    def test_non_callable_raises_type_error(self):
        with pytest.raises(TypeError, match="callable"):
            AttitudeLaw.custom("not_a_function")

    def test_pointing_eci_identity_returns_body_z(self):
        """Identity quaternions → boresight is always ECI body-z [0,0,1]."""
        law = AttitudeLaw.custom(_identity_cb)
        r, v, t = _orbit_state()
        p = law.pointing_eci(r, v, t)
        np.testing.assert_allclose(p, np.tile([0., 0., 1.], (len(t), 1)), atol=1e-12)

    def test_pointing_eci_returns_unit_vectors(self):
        law = AttitudeLaw.custom(_nadir_equiv_cb)
        r, v, t = _orbit_state()
        p = law.pointing_eci(r, v, t)
        norms = np.linalg.norm(p, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(t)), atol=1e-12)

    def test_rotate_from_body_identity_preserves_vector(self):
        """Identity quaternions → body-x stays ECI x-axis."""
        law = AttitudeLaw.custom(_identity_cb)
        r, v, t = _orbit_state()
        p = law.rotate_from_body([1., 0., 0.], r, v, t)
        np.testing.assert_allclose(p, np.tile([1., 0., 0.], (len(t), 1)), atol=1e-12)

    def test_rotate_from_body_z_matches_pointing_eci(self):
        """rotate_from_body([0,0,1]) must equal pointing_eci for any callback."""
        law = AttitudeLaw.custom(_nadir_equiv_cb)
        r, v, t = _orbit_state()
        boresight_via_pointing = law.pointing_eci(r, v, t)
        boresight_via_rotate   = law.rotate_from_body([0., 0., 1.], r, v, t)
        np.testing.assert_allclose(boresight_via_rotate, boresight_via_pointing, atol=1e-12)

    def test_callback_receives_correct_shapes(self):
        received = {}

        def recording_cb(t, r_eci, v_eci):
            received['t_shape'] = t.shape
            received['r_shape'] = r_eci.shape
            received['v_shape'] = v_eci.shape
            N = len(t)
            q = np.zeros((N, 4))
            q[:, 0] = 1.0
            return q

        law = AttitudeLaw.custom(recording_cb)
        r, v, t = _orbit_state()   # N=2
        law.pointing_eci(r, v, t)
        assert received['t_shape'] == (2,)
        assert received['r_shape'] == (2, 3)
        assert received['v_shape'] == (2, 3)

    def test_scalar_input_returns_1d(self):
        """Single-timestep (scalar) inputs must return (3,) shaped outputs."""
        law = AttitudeLaw.custom(_identity_cb)
        r, v, t = _orbit_state(_T2[:1])
        p_eci  = law.pointing_eci(r[0], v[0], t[0])
        p_rot  = law.rotate_from_body([0., 1., 0.], r[0], v[0], t[0])
        assert p_eci.shape == (3,)
        assert p_rot.shape == (3,)

    def test_yaw_steering_raises_not_implemented(self):
        from missiontools import NormalVectorSolarConfig
        law = AttitudeLaw.custom(_identity_cb)
        cfg = NormalVectorSolarConfig(normal_vecs=[[1, 0, 0]], areas=[1.0], efficiency=0.3)
        with pytest.raises(NotImplementedError):
            law.yaw_steering(cfg)

    def test_repr_contains_mode_and_callback_name(self):
        law = AttitudeLaw.custom(_identity_cb)
        r = repr(law)
        assert "custom" in r
        assert "_identity_cb" in r
