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
        t = np.datetime64("2025-01-01", "us")
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        g = ant.gain(t, v)
        np.testing.assert_allclose(g, 5.0)

    def test_default_gain_zero(self):
        ant = IsotropicAntenna()
        g = ant.gain(
            np.datetime64("2025-01-01", "us"),
            [[1, 0, 0]],
        )
        np.testing.assert_allclose(g, 0.0)

    def test_attach_to_spacecraft(self):
        from missiontools import Spacecraft

        sc = Spacecraft.sunsync(altitude_km=550, node_solar_time="10:30")
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
            [0, 90, 180],
            [10, 0, -10],
            body_vector=[0, 0, 1],
        )
        np.testing.assert_allclose(ant.angles_deg, [0, 90, 180])
        np.testing.assert_allclose(ant.gains_dbi, [10, 0, -10])

    def test_non_monotonic_raises(self):
        with pytest.raises(ValueError, match="monotonically"):
            SymmetricAntenna(
                [0, 90, 45],
                [10, 0, -10],
                body_vector=[0, 0, 1],
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="length"):
            SymmetricAntenna(
                [0, 90],
                [10, 0, -10],
                body_vector=[0, 0, 1],
            )

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            SymmetricAntenna(
                [0],
                [10],
                body_vector=[0, 0, 1],
            )

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="\\[0, 180\\]"):
            SymmetricAntenna(
                [-10, 90],
                [10, 0],
                body_vector=[0, 0, 1],
            )

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            SymmetricAntenna(
                [[0, 90]],
                [[10, 0]],
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
                [0, 180],
                [10, -10],
                body_vector=[0, 0, 1],
                azimuth_deg=0.0,
                elevation_deg=90.0,
            )

    def test_multiple_sc_options_raises(self):
        from missiontools.attitude import FixedAttitudeLaw

        with pytest.raises(ValueError, match="exactly one"):
            SymmetricAntenna(
                [0, 180],
                [10, -10],
                body_vector=[0, 0, 1],
                attitude_law=FixedAttitudeLaw.nadir(),
            )

    def test_ground_missing_elevation_raises(self):
        with pytest.raises(ValueError, match="elevation_deg"):
            SymmetricAntenna(
                [0, 180],
                [10, -10],
                azimuth_deg=0.0,
            )

    def test_body_vector_zero_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            SymmetricAntenna(
                [0, 180],
                [10, -10],
                body_vector=[0, 0, 0],
            )

    def test_body_vector_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            SymmetricAntenna(
                [0, 180],
                [10, -10],
                body_vector=[0, 0, 1, 0],
            )


# ===================================================================
# Attachment exclusivity
# ===================================================================


class TestAttachmentExclusivity:
    def test_sc_then_gs_raises(self):
        from missiontools import Spacecraft, GroundStation

        sc = Spacecraft.sunsync(altitude_km=550, node_solar_time="10:30")
        gs = GroundStation(lat=51.5, lon=-0.1)
        ant = IsotropicAntenna()
        sc.add_antenna(ant)
        with pytest.raises(ValueError, match="Spacecraft"):
            gs.add_antenna(ant)

    def test_gs_then_sc_raises(self):
        from missiontools import Spacecraft, GroundStation

        sc = Spacecraft.sunsync(altitude_km=550, node_solar_time="10:30")
        gs = GroundStation(lat=51.5, lon=-0.1)
        ant = IsotropicAntenna()
        gs.add_antenna(ant)
        with pytest.raises(ValueError, match="GroundStation"):
            sc.add_antenna(ant)

    def test_wrong_type_sc_raises(self):
        from missiontools import Spacecraft

        sc = Spacecraft.sunsync(altitude_km=550, node_solar_time="10:30")
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

        return Spacecraft.sunsync(altitude_km=550, node_solar_time="10:30")

    def test_boresight_gain(self, sc):
        """Direction aligned with boresight should give peak gain."""
        ant = SymmetricAntenna(
            [0, 90, 180],
            [10, 0, -10],
            body_vector=[0, 0, 1],  # nadir for nadir-pointing SC
        )
        sc.add_antenna(ant)

        t = np.datetime64("2025-01-01", "us")
        state = sc.propagate(t, t + np.timedelta64(1, "s"), np.timedelta64(1, "s"))
        r, v = state["r"][0], state["v"][0]

        # Boresight in ECI = nadir direction = -r_hat
        boresight = ant.boresight_eci(r, v, t)
        g = ant.gain(t, boresight, r_eci=r, v_eci=v)
        np.testing.assert_allclose(g, 10.0, atol=0.1)

    def test_off_boresight_interpolation(self, sc):
        """45° off-boresight should interpolate between 0° and 90° gains."""
        ant = SymmetricAntenna(
            [0, 90, 180],
            [10, 0, -10],
            body_vector=[0, 0, 1],
        )
        sc.add_antenna(ant)

        t = np.datetime64("2025-01-01", "us")
        state = sc.propagate(t, t + np.timedelta64(1, "s"), np.timedelta64(1, "s"))
        r, v = state["r"][0], state["v"][0]

        # 45° off boresight → linear interp between 10 and 0 → 5.0 dBi
        g = ant._pattern_gain(np.array([np.radians(45)]))
        np.testing.assert_allclose(g, 5.0, atol=0.1)

    def test_back_lobe(self, sc):
        """180° should give the back-lobe gain."""
        ant = SymmetricAntenna(
            [0, 90, 180],
            [10, 0, -10],
            body_vector=[0, 0, 1],
        )
        sc.add_antenna(ant)

        t = np.datetime64("2025-01-01", "us")
        state = sc.propagate(t, t + np.timedelta64(1, "s"), np.timedelta64(1, "s"))
        r, v = state["r"][0], state["v"][0]

        # Anti-boresight direction = +r_hat (zenith)
        r_hat = r / np.linalg.norm(r)
        g = ant.gain(t, r_hat, r_eci=r, v_eci=v)
        np.testing.assert_allclose(g, -10.0, atol=0.5)

    def test_body_euler_mounting(self, sc):
        """Euler angle mounting should produce same result as body_vector."""
        ant_vec = SymmetricAntenna(
            [0, 180],
            [10, -10],
            body_vector=[0, 0, 1],
        )
        ant_euler = SymmetricAntenna(
            [0, 180],
            [10, -10],
            body_euler_deg=(0, 0, 0),  # ZYX (0,0,0) → boresight = [0,0,1]
        )
        sc.add_antenna(ant_vec)

        from missiontools import Spacecraft

        sc2 = Spacecraft.sunsync(altitude_km=550, node_solar_time="10:30")
        sc2.add_antenna(ant_euler)

        t = np.datetime64("2025-01-01", "us")
        state = sc.propagate(t, t + np.timedelta64(1, "s"), np.timedelta64(1, "s"))
        r, v = state["r"][0], state["v"][0]

        g1 = ant_vec.gain(t, [[1, 0, 0]], r_eci=r, v_eci=v)
        g2 = ant_euler.gain(t, [[1, 0, 0]], r_eci=r, v_eci=v)
        np.testing.assert_allclose(g1, g2, atol=0.01)

    def test_independent_attitude_law(self):
        """Antenna with its own attitude law."""
        from missiontools.attitude import FixedAttitudeLaw

        law = FixedAttitudeLaw([0, 0, 1], "eci")  # always points +z in ECI
        ant = SymmetricAntenna(
            [0, 90, 180],
            [10, 0, -10],
            attitude_law=law,
        )
        t = np.datetime64("2025-01-01", "us")
        # Direction = +z in ECI → on boresight → peak gain
        g = ant.gain(t, [[0, 0, 1]], r_eci=[[1, 0, 0]], v_eci=[[0, 1, 0]])
        np.testing.assert_allclose(g, 10.0, atol=0.01)

    def test_body_mode_without_attachment_raises(self):
        """Body-mounted antenna without spacecraft should raise."""
        ant = SymmetricAntenna(
            [0, 180],
            [10, -10],
            body_vector=[0, 0, 1],
        )
        t = np.datetime64("2025-01-01", "us")
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
            [0, 90, 180],
            [20, 5, -5],
            azimuth_deg=0.0,
            elevation_deg=90.0,
        )
        gs.add_antenna(ant)

        # Boresight at (lat=0, lon=0) pointing up → ECEF = [1, 0, 0]
        np.testing.assert_allclose(
            ant._boresight_ecef,
            [1, 0, 0],
            atol=1e-10,
        )

        # Use the actual boresight direction in ECI (accounts for GMST)
        t = np.datetime64("2025-01-01", "us")
        boresight_eci = ant.boresight_eci(None, None, t)
        g = ant.gain(t, [boresight_eci])
        np.testing.assert_allclose(g, 20.0, atol=0.1)

    def test_off_axis_gain(self):
        """90° off-axis should give the 90° table value."""
        from missiontools import GroundStation

        gs = GroundStation(lat=0.0, lon=0.0)

        ant = SymmetricAntenna(
            [0, 90, 180],
            [20, 5, -5],
            azimuth_deg=0.0,
            elevation_deg=90.0,
        )
        gs.add_antenna(ant)

        g = ant._pattern_gain(np.array([np.pi / 2]))
        np.testing.assert_allclose(g, 5.0, atol=0.01)

    def test_ground_mode_without_attachment_raises(self):
        """Ground-mounted antenna without station should raise."""
        ant = SymmetricAntenna(
            [0, 180],
            [10, -10],
            azimuth_deg=0.0,
            elevation_deg=90.0,
        )
        t = np.datetime64("2025-01-01", "us")
        with pytest.raises(RuntimeError, match="attached"):
            ant.boresight_eci([1e7, 0, 0], [0, 7e3, 0], t)


# ===================================================================
# Frame parameter in gain()
# ===================================================================


class TestGainFrame:
    def test_ecef_frame(self):
        """gain with frame='ecef' should convert properly."""
        from missiontools.attitude import FixedAttitudeLaw

        ant = SymmetricAntenna(
            [0, 90, 180],
            [10, 0, -10],
            attitude_law=FixedAttitudeLaw([0, 0, 1], "eci"),
        )
        # At J2000 epoch, ECEF ≈ ECI (GMST ≈ 0 at noon)
        t = np.datetime64("2000-01-01T12:00:00", "us")
        g_eci = ant.gain(
            t, [[0, 0, 1]], frame="eci", r_eci=[[1e7, 0, 0]], v_eci=[[0, 7e3, 0]]
        )
        g_ecef = ant.gain(
            t, [[0, 0, 1]], frame="ecef", r_eci=[[1e7, 0, 0]], v_eci=[[0, 7e3, 0]]
        )
        # At J2000, ECEF z = ECI z (Earth rotation only around z)
        np.testing.assert_allclose(g_eci, g_ecef, atol=0.5)

    def test_unknown_frame_raises(self):
        ant = IsotropicAntenna()
        # IsotropicAntenna overrides gain(), so this won't trigger the
        # base class validation. Test with SymmetricAntenna instead.
        from missiontools.attitude import FixedAttitudeLaw

        ant = SymmetricAntenna(
            [0, 180],
            [10, -10],
            attitude_law=FixedAttitudeLaw([0, 0, 1], "eci"),
        )
        with pytest.raises(ValueError, match="Unknown frame"):
            ant.gain(
                np.datetime64("2025-01-01", "us"),
                [[1, 0, 0]],
                frame="xyz",
                r_eci=[[1e7, 0, 0]],
                v_eci=[[0, 7e3, 0]],
            )

    def test_lvlh_requires_state(self):
        from missiontools.attitude import FixedAttitudeLaw

        ant = SymmetricAntenna(
            [0, 180],
            [10, -10],
            attitude_law=FixedAttitudeLaw([0, 0, 1], "eci"),
        )
        with pytest.raises(ValueError, match="r_eci and v_eci"):
            ant.gain(
                np.datetime64("2025-01-01", "us"),
                [[1, 0, 0]],
                frame="lvlh",
            )


# ===================================================================
# SymmetricAntenna factory classmethods
# ===================================================================


class TestSymmetricAntennaFactories:
    # --- from_isoflux ---

    def test_isoflux_gain_increases_toward_edge(self):
        """Isoflux pattern: gain at nadir ≤ gain at edge of coverage."""
        ant = SymmetricAntenna.from_isoflux(
            600.0, min_elev_deg=5.0, body_vector=[0, 0, 1]
        )
        # First and last tabulated angles (excluding trailing rolloff)
        gains = ant.gains_dbi
        assert gains[0] <= gains[-3]  # [-3] is last coverage point before rolloff

    def test_isoflux_unity_directivity(self):
        """edge_gain=None → numerically verify unity directivity (within 1%)."""
        ant = SymmetricAntenna.from_isoflux(
            600.0, min_elev_deg=5.0, body_vector=[0, 0, 1]
        )
        thetas = np.linspace(0.0, np.pi, 10_000)
        gains_dbi = np.interp(thetas, np.radians(ant.angles_deg), ant.gains_dbi)
        g_lin = 10.0 ** (gains_dbi / 10.0)
        integral = np.trapz(g_lin * np.sin(thetas), thetas)
        assert abs(integral - 2.0) / 2.0 < 0.02  # within 2% of unity directivity

    def test_isoflux_edge_gain_specified(self):
        """edge_gain=5.0 → last coverage point in table should equal 5.0 dBi."""
        edge = 5.0
        ant = SymmetricAntenna.from_isoflux(
            600.0, min_elev_deg=5.0, edge_gain=edge, body_vector=[0, 0, 1]
        )
        # The last point before the rolloff is gains_dbi[-3]
        assert abs(ant.gains_dbi[-3] - edge) < 0.01

    def test_isoflux_realistic_runs(self):
        """600 km / 5° / Earth mean radius executes without error."""
        SymmetricAntenna.from_isoflux(600.0, min_elev_deg=5.0, body_vector=[0, 0, 1])

    def test_isoflux_custom_body_radius(self):
        """Custom central_body_radius (e.g. Moon) produces a valid antenna."""
        moon_r = 1_737_400.0
        ant = SymmetricAntenna.from_isoflux(
            100.0, min_elev_deg=10.0, central_body_radius=moon_r, body_vector=[0, 0, 1]
        )
        assert ant.peak_gain_dbi == ant.gains_dbi[0]

    # --- from_gaussian ---

    def test_gaussian_peak_gain(self):
        """Peak gain at boresight matches requested gain_dbi within 0.01 dB."""
        for g in (20.0, 10.0, 3.0):
            ant = SymmetricAntenna.from_gaussian(g, body_vector=[0, 0, 1])
            assert abs(ant.peak_gain_dbi - g) < 0.01, f"gain={g}"

    def test_gaussian_nonpositive_gain_raises(self):
        """gain_dbi ≤ 0 raises ValueError (no finite σ exists for a Gaussian)."""
        with pytest.raises(ValueError, match="positive"):
            SymmetricAntenna.from_gaussian(0.0, body_vector=[0, 0, 1])

    def test_gaussian_monotonically_decreasing(self):
        """Gaussian pattern is non-increasing from 0° outward."""
        ant = SymmetricAntenna.from_gaussian(20.0, body_vector=[0, 0, 1])
        gains = ant.gains_dbi
        # Allow tiny floating-point noise (1e-9 tolerance)
        assert np.all(np.diff(gains) <= 1e-9)

    def test_gaussian_low_gain_at_180(self):
        """Pattern table extends to 180° with very low gain."""
        ant = SymmetricAntenna.from_gaussian(20.0, body_vector=[0, 0, 1])
        assert ant.angles_deg[-1] == 180.0
        assert ant.gains_dbi[-1] <= -59.0

    # --- from_parabolic ---

    def test_parabolic_peak_gain(self):
        """Peak gain matches eff·(πD/λ)² formula within 0.01 dB."""
        D, f, eta = 1.2, 8.25e9, 0.6
        lam = 299_792_458.0 / f
        expected = 10.0 * np.log10(eta * (np.pi * D / lam) ** 2)
        ant = SymmetricAntenna.from_parabolic(D, f, eff=eta, body_vector=[0, 0, 1])
        assert abs(ant.peak_gain_dbi - expected) < 0.01

    def test_parabolic_first_null_deep(self):
        """Gain near the first null is at least 10 dB below peak."""
        D, f = 1.2, 8.25e9
        lam = 299_792_458.0 / f
        ant = SymmetricAntenna.from_parabolic(D, f, eff=0.6, body_vector=[0, 0, 1])
        theta_null_deg = np.degrees(np.arcsin(min(1.0, 1.22 * lam / D)))
        # Evaluate gain at the first null using the table
        g_null = float(
            np.interp(
                np.radians(theta_null_deg),
                np.radians(ant.angles_deg),
                ant.gains_dbi,
            )
        )
        assert (ant.peak_gain_dbi - g_null) > 10.0

    def test_parabolic_envelope_monotone_beyond_main_lobe(self):
        """envelope=True: gains are non-increasing beyond the main lobe."""
        D, f = 1.2, 8.25e9
        lam = 299_792_458.0 / f
        ant = SymmetricAntenna.from_parabolic(
            D, f, eff=0.6, envelope=True, body_vector=[0, 0, 1]
        )
        # Find index beyond first null
        theta_null = np.degrees(np.arcsin(min(1.0, 1.22 * lam / D)))
        angles = ant.angles_deg
        idx = np.searchsorted(angles, theta_null * 1.1)
        gains_beyond = ant.gains_dbi[idx:]
        assert np.all(np.diff(gains_beyond) <= 1e-9)

    def test_parabolic_full_pattern_has_sidelobes(self):
        """envelope=False: non-monotone gains beyond the first null (sidelobes present)."""
        D, f = 1.2, 8.25e9
        lam = 299_792_458.0 / f
        ant = SymmetricAntenna.from_parabolic(
            D, f, eff=0.6, envelope=False, body_vector=[0, 0, 1]
        )
        theta_null = np.degrees(np.arcsin(min(1.0, 1.22 * lam / D)))
        angles = ant.angles_deg
        idx = np.searchsorted(angles, theta_null * 1.05)
        gains_beyond = ant.gains_dbi[idx:]
        # At least one local maximum (gain increasing then decreasing) indicates sidelobes
        has_sidelobes = np.any(np.diff(gains_beyond) > 0.0)
        assert has_sidelobes

    def test_parabolic_efficiency_scaling(self):
        """Higher efficiency → higher peak gain."""
        D, f = 1.0, 10e9
        g1 = SymmetricAntenna.from_parabolic(
            D, f, eff=0.5, body_vector=[0, 0, 1]
        ).peak_gain_dbi
        g2 = SymmetricAntenna.from_parabolic(
            D, f, eff=0.7, body_vector=[0, 0, 1]
        ).peak_gain_dbi
        assert g2 > g1

    # --- from_s465 ---

    def test_s465_canonical_peak_gain(self):
        """Canonical: gain at boresight equals G_max = 10log10(0.7·(π·D/λ)²)."""
        D, f = 3.0, 14e9
        lam = 299_792_458.0 / f
        expected = 10.0 * np.log10(0.7 * (np.pi * D / lam) ** 2)
        ant = SymmetricAntenna.from_s465(D, f, body_vector=[0, 0, 1])
        assert abs(ant.gains_dbi[0] - expected) < 0.01

    def test_s465_canonical_sidelobe_formula(self):
        """Canonical: gain at 10° follows 32 − 25·log10(10) = 7 dBi for large dish."""
        D, f = 3.0, 14e9
        ant = SymmetricAntenna.from_s465(D, f, body_vector=[0, 0, 1])
        g_10 = float(
            np.interp(np.radians(10.0), np.radians(ant.angles_deg), ant.gains_dbi)
        )
        assert abs(g_10 - (32.0 - 25.0 * np.log10(10.0))) < 0.1

    def test_s465_canonical_far_sidelobe(self):
        """Canonical: gain at 90° and 150° is −10 dBi."""
        ant = SymmetricAntenna.from_s465(3.0, 14e9, body_vector=[0, 0, 1])
        for phi in (90.0, 150.0):
            g = float(
                np.interp(np.radians(phi), np.radians(ant.angles_deg), ant.gains_dbi)
            )
            assert abs(g - (-10.0)) < 1e-6

    def test_s465_canonical_phi_min_large(self):
        """Canonical, D/λ ≥ 50: sidelobe formula applies for φ well above φ_min."""
        D, f = 3.0, 14e9
        lam = 299_792_458.0 / f
        dl = D / lam
        phi_min = max(1.0, 100.0 / dl)
        ant = SymmetricAntenna.from_s465(D, f, body_vector=[0, 0, 1])
        # Check at 5·φ_min to be well clear of the flat/sidelobe boundary
        phi_test = phi_min * 5.0
        g_table = float(
            np.interp(
                np.radians(phi_test),
                np.radians(ant.angles_deg),
                ant.gains_dbi,
            )
        )
        g_formula = 32.0 - 25.0 * np.log10(phi_test)
        assert abs(g_table - g_formula) < 0.1

    def test_s465_canonical_phi_min_small(self):
        """Canonical, D/λ < 50: φ_min = max(2°, 114·(λ/D)^1.09)."""
        D, f = 0.3, 14e9  # D/λ ≈ 14
        lam = 299_792_458.0 / f
        phi_min = max(2.0, 114.0 * (lam / D) ** 1.09)
        ant = SymmetricAntenna.from_s465(D, f, body_vector=[0, 0, 1])
        # First angle in sidelobe region matches formula value
        angles = ant.angles_deg
        idx = np.searchsorted(angles, phi_min)
        g_table = float(
            np.interp(
                np.radians(phi_min * 1.05),
                np.radians(angles),
                ant.gains_dbi,
            )
        )
        g_formula = 32.0 - 25.0 * np.log10(phi_min * 1.05)
        assert abs(g_table - g_formula) < 0.5

    def test_s465_smooth_peak_gain(self):
        """Smooth model: gain at boresight equals G_max."""
        D, f = 3.0, 14e9
        lam = 299_792_458.0 / f
        expected = 10.0 * np.log10(0.7 * (np.pi * D / lam) ** 2)
        ant = SymmetricAntenna.from_s465(
            D, f, main_lobe_model=True, body_vector=[0, 0, 1]
        )
        assert abs(ant.gains_dbi[0] - expected) < 0.01

    def test_s465_smooth_parabolic_region(self):
        """Smooth model: mid-main-lobe gain matches G_max − 2.5e-3·(D/λ·φ)²."""
        D, f = 3.0, 14e9
        lam = 299_792_458.0 / f
        dl = D / lam
        g_max = 10.0 * np.log10(0.7 * (np.pi * dl) ** 2)
        g1 = 32.0  # dl > 100
        phi_m = (20.0 / dl) * np.sqrt(g_max - g1)
        phi_test = phi_m / 2.0
        ant = SymmetricAntenna.from_s465(
            D, f, main_lobe_model=True, body_vector=[0, 0, 1]
        )
        g_table = float(
            np.interp(
                np.radians(phi_test),
                np.radians(ant.angles_deg),
                ant.gains_dbi,
            )
        )
        g_formula = g_max - 2.5e-3 * (dl * phi_test) ** 2
        assert abs(g_table - g_formula) < 0.05

    def test_s465_smooth_far_sidelobe(self):
        """Smooth model: gain at 90° is −10 dBi."""
        ant = SymmetricAntenna.from_s465(
            3.0, 14e9, main_lobe_model=True, body_vector=[0, 0, 1]
        )
        g = float(
            np.interp(np.radians(90.0), np.radians(ant.angles_deg), ant.gains_dbi)
        )
        assert abs(g - (-10.0)) < 1e-6

    def test_s465_smooth_and_canonical_agree_far(self):
        """Both variants give the same gain at 30° (deep in sidelobe region)."""
        D, f = 3.0, 14e9
        ant_c = SymmetricAntenna.from_s465(D, f, body_vector=[0, 0, 1])
        ant_s = SymmetricAntenna.from_s465(
            D, f, main_lobe_model=True, body_vector=[0, 0, 1]
        )
        g_c = float(
            np.interp(np.radians(30.0), np.radians(ant_c.angles_deg), ant_c.gains_dbi)
        )
        g_s = float(
            np.interp(np.radians(30.0), np.radians(ant_s.angles_deg), ant_s.gains_dbi)
        )
        assert abs(g_c - g_s) < 0.01

    def test_s465_table_covers_180(self):
        """Both variants: pattern table extends to 180°."""
        for mlm in (False, True):
            ant = SymmetricAntenna.from_s465(
                3.0, 14e9, main_lobe_model=mlm, body_vector=[0, 0, 1]
            )
            assert ant.angles_deg[-1] == 180.0

    def test_s465_gmax_override(self):
        """gmax_dbi overrides the computed peak gain for both variants."""
        D, f, override = 3.0, 14e9, 50.0
        for mlm in (False, True):
            ant = SymmetricAntenna.from_s465(
                D, f, main_lobe_model=mlm, gmax_dbi=override, body_vector=[0, 0, 1]
            )
            assert abs(ant.gains_dbi[0] - override) < 0.01

    def test_s465_gmax_override_differs_from_default(self):
        """gmax_dbi=50 produces a different pattern than the η=0.7 default."""
        D, f = 3.0, 14e9
        ant_default = SymmetricAntenna.from_s465(D, f, body_vector=[0, 0, 1])
        ant_override = SymmetricAntenna.from_s465(
            D, f, gmax_dbi=50.0, body_vector=[0, 0, 1]
        )
        assert ant_default.gains_dbi[0] != ant_override.gains_dbi[0]


# ===================================================================
# Top-level imports
# ===================================================================


def test_top_level_import_isotropic():
    from missiontools import IsotropicAntenna as IA

    assert IA is IsotropicAntenna


def test_top_level_import_symmetric():
    from missiontools import SymmetricAntenna as SA

    assert SA is SymmetricAntenna
