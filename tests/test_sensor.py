import numpy as np
import pytest

from missiontools import (Spacecraft, FixedAttitudeLaw, TrackAttitudeLaw,
                          AbstractAttitudeLaw, Sensor)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_EPOCH = np.datetime64('2025-01-01T00:00:00', 'us')

_KW = dict(
    a      = 6_771_000.0,
    e      = 0.0,
    i      = np.radians(51.6),
    raan   = 0.0,
    arg_p  = 0.0,
    ma     = 0.0,
    epoch  = _EPOCH,
)


def _sc():
    return Spacecraft(**_KW)


def _orbit_state(sc):
    """Return a single ECI state (r, v, t) for testing pointing methods."""
    state = sc.propagate(_EPOCH, _EPOCH + np.timedelta64(60, 's'),
                         np.timedelta64(60, 's'))
    return state['r'][0], state['v'][0], state['t'][0]


# ===========================================================================
# Construction and validation
# ===========================================================================

class TestSensorConstruct:

    def test_no_mode_raises(self):
        with pytest.raises(ValueError, match='Exactly one'):
            Sensor(10.0)

    def test_two_modes_raises(self):
        with pytest.raises(ValueError, match='Only one'):
            Sensor(10.0,
                   attitude_law=FixedAttitudeLaw.nadir(),
                   body_vector=[0, 0, 1])

    def test_independent_mode_stored(self):
        law = FixedAttitudeLaw.nadir()
        s = Sensor(10.0, attitude_law=law)
        assert s._mode == 'independent'
        assert s._attitude_law is law

    def test_body_vector_mode_stored(self):
        s = Sensor(10.0, body_vector=[0, 0, 1])
        assert s._mode == 'body'

    def test_body_euler_mode_stored(self):
        s = Sensor(10.0, body_euler_deg=(0, 0, 0))
        assert s._mode == 'body'

    def test_invalid_attitude_law_type_raises(self):
        with pytest.raises(TypeError, match='AbstractAttitudeLaw'):
            Sensor(10.0, attitude_law='nadir')


class TestSensorHalfAngle:

    def test_stored_as_radians(self):
        s = Sensor(30.0, body_vector=[0, 0, 1])
        np.testing.assert_allclose(s.half_angle_rad, np.radians(30.0))

    def test_90_deg_accepted(self):
        s = Sensor(90.0, body_vector=[0, 0, 1])
        np.testing.assert_allclose(s.half_angle_rad, np.pi / 2)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match='half_angle_deg'):
            Sensor(0.0, body_vector=[0, 0, 1])

    def test_negative_raises(self):
        with pytest.raises(ValueError, match='half_angle_deg'):
            Sensor(-5.0, body_vector=[0, 0, 1])

    def test_above_90_raises(self):
        with pytest.raises(ValueError, match='half_angle_deg'):
            Sensor(91.0, body_vector=[0, 0, 1])


# ===========================================================================
# Body-vector mode
# ===========================================================================

class TestSensorBodyVector:

    def test_body_vector_is_unit(self):
        s = Sensor(10.0, body_vector=[3, 0, 0])
        np.testing.assert_allclose(np.linalg.norm(s._body_vector), 1.0)

    def test_body_vector_non_unit_normalised(self):
        s = Sensor(10.0, body_vector=[0, 0, 5])
        np.testing.assert_allclose(s._body_vector, [0, 0, 1])

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match='zero vector'):
            Sensor(10.0, body_vector=[0, 0, 0])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match='shape'):
            Sensor(10.0, body_vector=[1, 0])

    def test_pointing_raises_before_attach(self):
        s = Sensor(10.0, body_vector=[0, 0, 1])
        sc = _sc()
        r, v, t = _orbit_state(sc)
        with pytest.raises(RuntimeError, match='add_sensor'):
            s.pointing_eci(r, v, t)

    def test_spacecraft_none_before_attach(self):
        s = Sensor(10.0, body_vector=[0, 0, 1])
        assert s.spacecraft is None


# ===========================================================================
# Body-euler mode
# ===========================================================================

class TestSensorBodyEuler:

    def test_identity_euler_gives_body_z(self):
        s = Sensor(10.0, body_euler_deg=(0, 0, 0))
        np.testing.assert_allclose(s._body_vector, [0, 0, 1], atol=1e-12)

    def test_pitch_90_gives_negative_body_x(self):
        # 90° pitch (yaw=0, pitch=90, roll=0): sensor-z → body-x neg direction
        s = Sensor(10.0, body_euler_deg=(0, 90, 0))
        np.testing.assert_allclose(s._body_vector, [-1, 0, 0], atol=1e-12)

    def test_equivalent_to_body_vector(self):
        # Identity euler → same as body_vector=[0,0,1]
        s_euler  = Sensor(10.0, body_euler_deg=(0, 0, 0))
        s_vector = Sensor(10.0, body_vector=[0, 0, 1])
        np.testing.assert_allclose(s_euler._body_vector, s_vector._body_vector,
                                   atol=1e-12)

    def test_yaw_only_does_not_change_boresight(self):
        # Yaw rotates about body-z — does not move the z-axis
        s0 = Sensor(10.0, body_euler_deg=(0,   0, 0))
        s1 = Sensor(10.0, body_euler_deg=(45,  0, 0))
        s2 = Sensor(10.0, body_euler_deg=(90,  0, 0))
        np.testing.assert_allclose(s0._body_vector, s1._body_vector, atol=1e-12)
        np.testing.assert_allclose(s0._body_vector, s2._body_vector, atol=1e-12)


# ===========================================================================
# Independent mode
# ===========================================================================

class TestSensorIndependent:

    def test_stores_attitude_law(self):
        law = FixedAttitudeLaw.nadir()
        s = Sensor(10.0, attitude_law=law)
        assert s._attitude_law is law

    def test_pointing_eci_delegates(self):
        law = FixedAttitudeLaw.nadir()
        s   = Sensor(10.0, attitude_law=law)
        sc  = _sc()
        r, v, t = _orbit_state(sc)
        np.testing.assert_allclose(
            s.pointing_eci(r, v, t),
            law.pointing_eci(r, v, t),
            atol=1e-12,
        )

    def test_pointing_lvlh_delegates(self):
        law = FixedAttitudeLaw.nadir()
        s   = Sensor(10.0, attitude_law=law)
        sc  = _sc()
        r, v, t = _orbit_state(sc)
        np.testing.assert_allclose(
            s.pointing_lvlh(r, v, t),
            law.pointing_lvlh(r, v, t),
            atol=1e-12,
        )

    def test_pointing_ecef_delegates(self):
        law = FixedAttitudeLaw.nadir()
        s   = Sensor(10.0, attitude_law=law)
        sc  = _sc()
        r, v, t = _orbit_state(sc)
        np.testing.assert_allclose(
            s.pointing_ecef(r, v, t),
            law.pointing_ecef(r, v, t),
            atol=1e-12,
        )

    def test_no_spacecraft_needed(self):
        # Independent sensors work without an attached spacecraft
        s = Sensor(10.0, attitude_law=FixedAttitudeLaw.nadir())
        assert s.spacecraft is None
        sc = _sc()
        r, v, t = _orbit_state(sc)
        result = s.pointing_eci(r, v, t)
        assert result.shape == (3,)


# ===========================================================================
# Spacecraft–sensor relationship
# ===========================================================================

class TestSpacecraftSensorRelationship:

    def test_sensors_empty_by_default(self):
        assert _sc().sensors == []

    def test_add_sensor_grows_list(self):
        sc = _sc()
        s  = Sensor(10.0, body_vector=[0, 0, 1])
        sc.add_sensor(s)
        assert len(sc.sensors) == 1

    def test_back_reference_set(self):
        sc = _sc()
        s  = Sensor(10.0, body_vector=[0, 0, 1])
        sc.add_sensor(s)
        assert s.spacecraft is sc

    def test_sensors_returns_copy(self):
        sc = _sc()
        sc.add_sensor(Sensor(10.0, body_vector=[0, 0, 1]))
        lst = sc.sensors
        lst.clear()
        assert len(sc.sensors) == 1    # original unaffected

    def test_multiple_sensors_stored(self):
        sc = _sc()
        s1 = Sensor(10.0, body_vector=[0, 0, 1])
        s2 = Sensor(20.0, body_vector=[0, 1, 0])
        sc.add_sensor(s1)
        sc.add_sensor(s2)
        assert len(sc.sensors) == 2
        assert sc.sensors[0] is s1
        assert sc.sensors[1] is s2

    def test_wrong_type_raises(self):
        sc = _sc()
        with pytest.raises(TypeError, match='Sensor'):
            sc.add_sensor('not_a_sensor')


# ===========================================================================
# Pointing correctness
# ===========================================================================

class TestSensorPointing:
    """Verify body-mounted sensor pointing directions on a nadir spacecraft.

    Nadir spacecraft (FixedAttitudeLaw.nadir()) quaternion maps body frame → LVLH:
      body-z [0,0,1] → LVLH [-1, 0, 0]  (nadir = −R̂)
      body-x [1,0,0] → LVLH [ 0, 1, 0]  (along-track = Ŝ)
      body-y [0,1,0] → LVLH [ 0, 0,-1]  (−orbit-normal = −Ŵ)
    """

    def _setup(self, body_vec):
        sc = _sc()                                # nadir spacecraft by default
        s  = Sensor(10.0, body_vector=body_vec)
        sc.add_sensor(s)
        r, v, t = _orbit_state(sc)
        return s, r, v, t

    def test_body_z_sensor_on_nadir_sc_points_nadir(self):
        s, r, v, t = self._setup([0, 0, 1])
        lvlh = s.pointing_lvlh(r, v, t)
        np.testing.assert_allclose(lvlh, [-1., 0., 0.], atol=1e-10)

    def test_body_x_sensor_on_nadir_sc_points_along_track(self):
        s, r, v, t = self._setup([1, 0, 0])
        lvlh = s.pointing_lvlh(r, v, t)
        np.testing.assert_allclose(lvlh, [0., 1., 0.], atol=1e-10)

    def test_body_y_sensor_on_nadir_sc_points_minus_orbit_normal(self):
        s, r, v, t = self._setup([0, 1, 0])
        lvlh = s.pointing_lvlh(r, v, t)
        np.testing.assert_allclose(lvlh, [0., 0., -1.], atol=1e-10)

    def test_pointing_eci_is_unit(self):
        s, r, v, t = self._setup([0, 0, 1])
        np.testing.assert_allclose(np.linalg.norm(s.pointing_eci(r, v, t)), 1.0,
                                   atol=1e-12)

    def test_pointing_lvlh_is_unit(self):
        s, r, v, t = self._setup([0, 1, 0])
        np.testing.assert_allclose(np.linalg.norm(s.pointing_lvlh(r, v, t)), 1.0,
                                   atol=1e-12)

    def test_pointing_ecef_is_unit(self):
        s, r, v, t = self._setup([1, 0, 0])
        np.testing.assert_allclose(np.linalg.norm(s.pointing_ecef(r, v, t)), 1.0,
                                   atol=1e-12)

    def test_body_mounted_sensor_on_tracking_sc_boresight_matches(self):
        """Body-z sensor on a tracking spacecraft must point at the target."""
        sc_host   = _sc()
        sc_target = Spacecraft(
            a=7_000_000., e=0., i=np.radians(60.), raan=0.5,
            arg_p=0., ma=0., epoch=_EPOCH,
        )
        sc_host.attitude_law = TrackAttitudeLaw(sc_target)
        s = Sensor(10.0, body_vector=[0, 0, 1])   # sensor boresight = body-z
        sc_host.add_sensor(s)

        r, v, t = _orbit_state(sc_host)
        sensor_pointing = s.pointing_eci(r, v, t)
        law_pointing    = sc_host.attitude_law.pointing_eci(r, v, t)
        np.testing.assert_allclose(sensor_pointing, law_pointing, atol=1e-10)
