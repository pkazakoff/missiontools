"""Tests for missiontools.comm antenna classes."""

import numpy as np
import pytest

from missiontools.comm import AbstractAntenna, IsotropicAntenna, SymmetricAntenna
from missiontools.orbit.frames import azel_to_enu, enu_to_ecef


# ===================================================================
# Frame helpers
# ===================================================================

class TestAzelToEnu:
    def test_zenith(self):
        """Elevation 90° should point straight up."""
        v = azel_to_enu(0.0, np.pi / 2)
        np.testing.assert_allclose(v, [0, 0, 1], atol=1e-15)

    def test_north(self):
        """Azimuth 0°, elevation 0° should point north."""
        v = azel_to_enu(0.0, 0.0)
        np.testing.assert_allclose(v, [0, 1, 0], atol=1e-15)

    def test_east(self):
        """Azimuth 90°, elevation 0° should point east."""
        v = azel_to_enu(np.pi / 2, 0.0)
        np.testing.assert_allclose(v, [1, 0, 0], atol=1e-15)

    def test_south(self):
        """Azimuth 180°, elevation 0° should point south."""
        v = azel_to_enu(np.pi, 0.0)
        np.testing.assert_allclose(v, [0, -1, 0], atol=1e-15)

    def test_unit_vector(self):
        """Result should be a unit vector for any input."""
        v = azel_to_enu(np.radians(45), np.radians(30))
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-15)


class TestEnuToEcef:
    def test_up_at_equator_prime_meridian(self):
        """Up at (lat=0, lon=0) should align with ECEF x-axis."""
        v_enu = np.array([0.0, 0.0, 1.0])
        v_ecef = enu_to_ecef(v_enu, lat=0.0, lon=0.0)
        np.testing.assert_allclose(v_ecef, [1, 0, 0], atol=1e-15)

    def test_north_at_equator_prime_meridian(self):
        """North at (lat=0, lon=0) should align with ECEF -z direction
        (pointing toward north pole from equator is +z, but 'north' in
        ENU at equator is tangent to surface → ECEF z)."""
        v_enu = np.array([0.0, 1.0, 0.0])
        v_ecef = enu_to_ecef(v_enu, lat=0.0, lon=0.0)
        np.testing.assert_allclose(v_ecef, [0, 0, 1], atol=1e-15)

    def test_east_at_equator_prime_meridian(self):
        """East at (lat=0, lon=0) should align with ECEF y."""
        v_enu = np.array([1.0, 0.0, 0.0])
        v_ecef = enu_to_ecef(v_enu, lat=0.0, lon=0.0)
        np.testing.assert_allclose(v_ecef, [0, 1, 0], atol=1e-15)

    def test_up_at_north_pole(self):
        """Up at (lat=90°, lon=0) should align with ECEF z."""
        v_enu = np.array([0.0, 0.0, 1.0])
        v_ecef = enu_to_ecef(v_enu, lat=np.pi / 2, lon=0.0)
        np.testing.assert_allclose(v_ecef, [0, 0, 1], atol=1e-15)

    def test_batch(self):
        """Batch of vectors should work."""
        vecs = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        result = enu_to_ecef(vecs, lat=0.0, lon=0.0)
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result[0], [1, 0, 0], atol=1e-15)

    def test_scalar_returns_1d(self):
        """Scalar input should return shape (3,)."""
        v = enu_to_ecef([0, 0, 1], lat=0.0, lon=0.0)
        assert v.shape == (3,)


# ===================================================================
# IsotropicAntenna
# ===================================================================

class TestIsotropicAntenna:
    def test_constant_gain(self):
        ant = IsotropicAntenna(gain_dbi=5.0)
        t = np.datetime64('2025-01-01', 'us')
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        g = ant.gain(t, v)
        np.testing.assert_allclose(g, 5.0)

    def test_default_gain_zero(self):
        ant = IsotropicAntenna()
        g = ant.gain(
            np.datetime64('2025-01-01', 'us'),
            [[1, 0, 0]],
        )
        np.testing.assert_allclose(g, 0.0)

    def test_attach_to_spacecraft(self):
        from missiontools import Spacecraft
        sc = Spacecraft.sunsync(altitude_km=550, node_solar_time='10:30')
        ant = IsotropicAntenna(gain_dbi=3.0)
        sc.add_antenna(ant)
        assert ant.spacecraft is sc
        assert ant.host is sc
        assert len(sc.antennas) == 1

    def test_attach_to_ground_station(self):
        from missiontools import GroundStation
        gs = GroundStation(lat=51.5, lon=-0.1)
        ant = IsotropicAntenna(gain_dbi=3.0)
        gs.add_antenna(ant)
        assert ant.ground_station is gs
        assert ant.host is gs
        assert len(gs.antennas) == 1


# ===================================================================
# SymmetricAntenna construction
# ===================================================================

class TestSymmetricAntennaConstruction:
    def test_basic(self):
        ant = SymmetricAntenna(
            [0, 90, 180], [10, 0, -10],
            body_vector=[0, 0, 1],
        )
        np.testing.assert_allclose(ant.angles_deg, [0, 90, 180])
        np.testing.assert_allclose(ant.gains_dbi, [10, 0, -10])

    def test_non_monotonic_raises(self):
        with pytest.raises(ValueError, match="monotonically"):
            SymmetricAntenna(
                [0, 90, 45], [10, 0, -10],
                body_vector=[0, 0, 1],
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="length"):
            SymmetricAntenna(
                [0, 90], [10, 0, -10],
                body_vector=[0, 0, 1],
            )

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            SymmetricAntenna(
                [0], [10],
                body_vector=[0, 0, 1],
            )

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="\\[0, 180\\]"):
            SymmetricAntenna(
                [-10, 90], [10, 0],
                body_vector=[0, 0, 1],
            )

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            SymmetricAntenna(
                [[0, 90]], [[10, 0]],
                body_vector=[0, 0, 1],
            )


# ===================================================================
# Mounting validation
# ===================================================================

class TestMounting:
    def test_no_mounting_raises(self):
        """AbstractAntenna (via SymmetricAntenna) requires mounting."""
        with pytest.raises(ValueError, match="Must specify"):
            SymmetricAntenna([0, 180], [10, -10])

    def test_mixed_mounting_raises(self):
        with pytest.raises(ValueError, match="Cannot mix"):
            SymmetricAntenna(
                [0, 180], [10, -10],
                body_vector=[0, 0, 1],
                azimuth_deg=0.0,
                elevation_deg=90.0,
            )

    def test_multiple_sc_options_raises(self):
        from missiontools.attitude import AttitudeLaw
        with pytest.raises(ValueError, match="exactly one"):
            SymmetricAntenna(
                [0, 180], [10, -10],
                body_vector=[0, 0, 1],
                attitude_law=AttitudeLaw.nadir(),
            )

    def test_ground_missing_elevation_raises(self):
        with pytest.raises(ValueError, match="elevation_deg"):
            SymmetricAntenna(
                [0, 180], [10, -10],
                azimuth_deg=0.0,
            )

    def test_body_vector_zero_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            SymmetricAntenna(
                [0, 180], [10, -10],
                body_vector=[0, 0, 0],
            )

    def test_body_vector_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            SymmetricAntenna(
                [0, 180], [10, -10],
                body_vector=[0, 0, 1, 0],
            )


# ===================================================================
# Attachment exclusivity
# ===================================================================

class TestAttachmentExclusivity:
    def test_sc_then_gs_raises(self):
        from missiontools import Spacecraft, GroundStation
        sc = Spacecraft.sunsync(altitude_km=550, node_solar_time='10:30')
        gs = GroundStation(lat=51.5, lon=-0.1)
        ant = IsotropicAntenna()
        sc.add_antenna(ant)
        with pytest.raises(ValueError, match="Spacecraft"):
            gs.add_antenna(ant)

    def test_gs_then_sc_raises(self):
        from missiontools import Spacecraft, GroundStation
        sc = Spacecraft.sunsync(altitude_km=550, node_solar_time='10:30')
        gs = GroundStation(lat=51.5, lon=-0.1)
        ant = IsotropicAntenna()
        gs.add_antenna(ant)
        with pytest.raises(ValueError, match="GroundStation"):
            sc.add_antenna(ant)

    def test_wrong_type_sc_raises(self):
        from missiontools import Spacecraft
        sc = Spacecraft.sunsync(altitude_km=550, node_solar_time='10:30')
        with pytest.raises(TypeError, match="AbstractAntenna"):
            sc.add_antenna("not an antenna")

    def test_wrong_type_gs_raises(self):
        from missiontools import GroundStation
        gs = GroundStation(lat=51.5, lon=-0.1)
        with pytest.raises(TypeError, match="AbstractAntenna"):
            gs.add_antenna("not an antenna")


# ===================================================================
# SymmetricAntenna gain — spacecraft mounted
# ===================================================================

class TestSymmetricAntennaSpacecraft:
    @pytest.fixture
    def sc(self):
        from missiontools import Spacecraft
        return Spacecraft.sunsync(altitude_km=550, node_solar_time='10:30')

    def test_boresight_gain(self, sc):
        """Direction aligned with boresight should give peak gain."""
        ant = SymmetricAntenna(
            [0, 90, 180], [10, 0, -10],
            body_vector=[0, 0, 1],  # nadir for nadir-pointing SC
        )
        sc.add_antenna(ant)

        t = np.datetime64('2025-01-01', 'us')
        state = sc.propagate(t, t + np.timedelta64(1, 's'),
                             np.timedelta64(1, 's'))
        r, v = state['r'][0], state['v'][0]

        # Boresight in ECI = nadir direction = -r_hat
        boresight = ant.boresight_eci(r, v, t)
        g = ant.gain(t, boresight, r_eci=r, v_eci=v)
        np.testing.assert_allclose(g, 10.0, atol=0.1)

    def test_off_boresight_interpolation(self, sc):
        """45° off-boresight should interpolate between 0° and 90° gains."""
        ant = SymmetricAntenna(
            [0, 90, 180], [10, 0, -10],
            body_vector=[0, 0, 1],
        )
        sc.add_antenna(ant)

        t = np.datetime64('2025-01-01', 'us')
        state = sc.propagate(t, t + np.timedelta64(1, 's'),
                             np.timedelta64(1, 's'))
        r, v = state['r'][0], state['v'][0]

        # 45° off boresight → linear interp between 10 and 0 → 5.0 dBi
        g = ant._pattern_gain(np.array([np.radians(45)]))
        np.testing.assert_allclose(g, 5.0, atol=0.1)

    def test_back_lobe(self, sc):
        """180° should give the back-lobe gain."""
        ant = SymmetricAntenna(
            [0, 90, 180], [10, 0, -10],
            body_vector=[0, 0, 1],
        )
        sc.add_antenna(ant)

        t = np.datetime64('2025-01-01', 'us')
        state = sc.propagate(t, t + np.timedelta64(1, 's'),
                             np.timedelta64(1, 's'))
        r, v = state['r'][0], state['v'][0]

        # Anti-boresight direction = +r_hat (zenith)
        r_hat = r / np.linalg.norm(r)
        g = ant.gain(t, r_hat, r_eci=r, v_eci=v)
        np.testing.assert_allclose(g, -10.0, atol=0.5)

    def test_body_euler_mounting(self, sc):
        """Euler angle mounting should produce same result as body_vector."""
        ant_vec = SymmetricAntenna(
            [0, 180], [10, -10],
            body_vector=[0, 0, 1],
        )
        ant_euler = SymmetricAntenna(
            [0, 180], [10, -10],
            body_euler_deg=(0, 0, 0),  # ZYX (0,0,0) → boresight = [0,0,1]
        )
        sc.add_antenna(ant_vec)

        from missiontools import Spacecraft
        sc2 = Spacecraft.sunsync(altitude_km=550, node_solar_time='10:30')
        sc2.add_antenna(ant_euler)

        t = np.datetime64('2025-01-01', 'us')
        state = sc.propagate(t, t + np.timedelta64(1, 's'),
                             np.timedelta64(1, 's'))
        r, v = state['r'][0], state['v'][0]

        g1 = ant_vec.gain(t, [[1, 0, 0]], r_eci=r, v_eci=v)
        g2 = ant_euler.gain(t, [[1, 0, 0]], r_eci=r, v_eci=v)
        np.testing.assert_allclose(g1, g2, atol=0.01)

    def test_independent_attitude_law(self):
        """Antenna with its own AttitudeLaw."""
        from missiontools.attitude import AttitudeLaw
        law = AttitudeLaw.fixed([0, 0, 1], 'eci')  # always points +z in ECI
        ant = SymmetricAntenna(
            [0, 90, 180], [10, 0, -10],
            attitude_law=law,
        )
        t = np.datetime64('2025-01-01', 'us')
        # Direction = +z in ECI → on boresight → peak gain
        g = ant.gain(t, [[0, 0, 1]], r_eci=[[1, 0, 0]], v_eci=[[0, 1, 0]])
        np.testing.assert_allclose(g, 10.0, atol=0.01)

    def test_body_mode_without_attachment_raises(self):
        """Body-mounted antenna without spacecraft should raise."""
        ant = SymmetricAntenna(
            [0, 180], [10, -10],
            body_vector=[0, 0, 1],
        )
        t = np.datetime64('2025-01-01', 'us')
        with pytest.raises(RuntimeError, match="attached"):
            ant.boresight_eci([1e7, 0, 0], [0, 7e3, 0], t)


# ===================================================================
# SymmetricAntenna gain — ground station mounted
# ===================================================================

class TestSymmetricAntennaGroundStation:
    def test_boresight_gain(self):
        """Gain along boresight should be peak gain."""
        from missiontools import GroundStation
        gs = GroundStation(lat=0.0, lon=0.0)

        # Zenith-pointing antenna at equator/prime meridian
        ant = SymmetricAntenna(
            [0, 90, 180], [20, 5, -5],
            azimuth_deg=0.0,
            elevation_deg=90.0,
        )
        gs.add_antenna(ant)

        # Boresight at (lat=0, lon=0) pointing up → ECEF = [1, 0, 0]
        np.testing.assert_allclose(
            ant._boresight_ecef, [1, 0, 0], atol=1e-10,
        )

        # Use the actual boresight direction in ECI (accounts for GMST)
        t = np.datetime64('2025-01-01', 'us')
        boresight_eci = ant.boresight_eci(None, None, t)
        g = ant.gain(t, [boresight_eci])
        np.testing.assert_allclose(g, 20.0, atol=0.1)

    def test_off_axis_gain(self):
        """90° off-axis should give the 90° table value."""
        from missiontools import GroundStation
        gs = GroundStation(lat=0.0, lon=0.0)

        ant = SymmetricAntenna(
            [0, 90, 180], [20, 5, -5],
            azimuth_deg=0.0,
            elevation_deg=90.0,
        )
        gs.add_antenna(ant)

        g = ant._pattern_gain(np.array([np.pi / 2]))
        np.testing.assert_allclose(g, 5.0, atol=0.01)

    def test_ground_mode_without_attachment_raises(self):
        """Ground-mounted antenna without station should raise."""
        ant = SymmetricAntenna(
            [0, 180], [10, -10],
            azimuth_deg=0.0,
            elevation_deg=90.0,
        )
        t = np.datetime64('2025-01-01', 'us')
        with pytest.raises(RuntimeError, match="attached"):
            ant.boresight_eci([1e7, 0, 0], [0, 7e3, 0], t)


# ===================================================================
# Frame parameter in gain()
# ===================================================================

class TestGainFrame:
    def test_ecef_frame(self):
        """gain with frame='ecef' should convert properly."""
        from missiontools.attitude import AttitudeLaw
        ant = SymmetricAntenna(
            [0, 90, 180], [10, 0, -10],
            attitude_law=AttitudeLaw.fixed([0, 0, 1], 'eci'),
        )
        # At J2000 epoch, ECEF ≈ ECI (GMST ≈ 0 at noon)
        t = np.datetime64('2000-01-01T12:00:00', 'us')
        g_eci = ant.gain(t, [[0, 0, 1]], frame='eci',
                         r_eci=[[1e7, 0, 0]], v_eci=[[0, 7e3, 0]])
        g_ecef = ant.gain(t, [[0, 0, 1]], frame='ecef',
                          r_eci=[[1e7, 0, 0]], v_eci=[[0, 7e3, 0]])
        # At J2000, ECEF z = ECI z (Earth rotation only around z)
        np.testing.assert_allclose(g_eci, g_ecef, atol=0.5)

    def test_unknown_frame_raises(self):
        ant = IsotropicAntenna()
        # IsotropicAntenna overrides gain(), so this won't trigger the
        # base class validation. Test with SymmetricAntenna instead.
        from missiontools.attitude import AttitudeLaw
        ant = SymmetricAntenna(
            [0, 180], [10, -10],
            attitude_law=AttitudeLaw.fixed([0, 0, 1], 'eci'),
        )
        with pytest.raises(ValueError, match="Unknown frame"):
            ant.gain(
                np.datetime64('2025-01-01', 'us'),
                [[1, 0, 0]], frame='xyz',
                r_eci=[[1e7, 0, 0]], v_eci=[[0, 7e3, 0]],
            )

    def test_lvlh_requires_state(self):
        from missiontools.attitude import AttitudeLaw
        ant = SymmetricAntenna(
            [0, 180], [10, -10],
            attitude_law=AttitudeLaw.fixed([0, 0, 1], 'eci'),
        )
        with pytest.raises(ValueError, match="r_eci and v_eci"):
            ant.gain(
                np.datetime64('2025-01-01', 'us'),
                [[1, 0, 0]], frame='lvlh',
            )


# ===================================================================
# Top-level imports
# ===================================================================

def test_top_level_import_isotropic():
    from missiontools import IsotropicAntenna as IA
    assert IA is IsotropicAntenna

def test_top_level_import_symmetric():
    from missiontools import SymmetricAntenna as SA
    assert SA is SymmetricAntenna
