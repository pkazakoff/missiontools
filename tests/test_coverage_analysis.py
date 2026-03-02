import numpy as np
import pytest

from missiontools import Spacecraft, AttitudeLaw, Sensor, AoI, Coverage

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_EPOCH = np.datetime64('2025-01-01T00:00:00', 'us')
_T_END = _EPOCH + np.timedelta64(3600, 's')   # 1 hour
_STEP  = np.timedelta64(300, 's')             # 5-min step (fast tests)

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


def _sensor():
    """Body-mounted nadir sensor (body-z boresight, 30° FOV)."""
    return Sensor(30.0, body_vector=[0, 0, 1])


def _aoi():
    """Small 20°×20° equatorial region for fast tests."""
    return AoI.from_region(-10, 10, -10, 10, point_density=5e12)


def _attached_sensor():
    """Ready-to-use sensor already attached to a fresh spacecraft."""
    sc = _sc()
    s  = _sensor()
    sc.add_sensor(s)
    return s


# ===========================================================================
# Construction and validation
# ===========================================================================

class TestCoverageConstruct:

    def test_valid_construction(self):
        s = _attached_sensor()
        Coverage(_aoi(), [s])   # must not raise

    def test_aoi_wrong_type_raises(self):
        s = _attached_sensor()
        with pytest.raises(TypeError, match='AoI'):
            Coverage('not_an_aoi', [s])

    def test_sensors_not_a_list_raises(self):
        s = _attached_sensor()
        with pytest.raises(ValueError, match='non-empty'):
            Coverage(_aoi(), s)   # bare Sensor, not a list

    def test_sensors_empty_list_raises(self):
        with pytest.raises(ValueError, match='non-empty'):
            Coverage(_aoi(), [])

    def test_sensor_wrong_element_type_raises(self):
        with pytest.raises(TypeError, match='Sensor'):
            Coverage(_aoi(), ['not_a_sensor'])

    def test_multiple_sensors_raises(self):
        sc = _sc()
        s1 = Sensor(10.0, body_vector=[0, 0, 1])
        s2 = Sensor(10.0, body_vector=[0, 1, 0])
        sc.add_sensor(s1)
        sc.add_sensor(s2)
        with pytest.raises(NotImplementedError, match='Multiple'):
            Coverage(_aoi(), [s1, s2])

    def test_unattached_sensor_raises(self):
        s = _sensor()   # not attached to any spacecraft
        with pytest.raises(RuntimeError, match='add_sensor'):
            Coverage(_aoi(), [s])

    def test_negative_el_min_raises(self):
        s = _attached_sensor()
        with pytest.raises(ValueError, match='el_min_deg'):
            Coverage(_aoi(), [s], el_min_deg=-1.0)

    def test_zero_el_min_accepted(self):
        s = _attached_sensor()
        Coverage(_aoi(), [s], el_min_deg=0.0)

    def test_positive_el_min_accepted(self):
        s = _attached_sensor()
        Coverage(_aoi(), [s], el_min_deg=5.0)

    def test_sza_constraints_accepted(self):
        s = _attached_sensor()
        Coverage(_aoi(), [s], sza_max_deg=70.0, sza_min_deg=10.0)

    def test_independent_sensor_attached_works(self):
        sc = _sc()
        s  = Sensor(30.0, attitude_law=AttitudeLaw.nadir())
        sc.add_sensor(s)
        Coverage(_aoi(), [s])   # must not raise


# ===========================================================================
# Properties
# ===========================================================================

class TestCoverageProperties:

    def test_aoi_property(self):
        aoi = _aoi()
        s   = _attached_sensor()
        cov = Coverage(aoi, [s])
        assert cov.aoi is aoi

    def test_sensors_property_returns_copy(self):
        s   = _attached_sensor()
        cov = Coverage(_aoi(), [s])
        lst = cov.sensors
        lst.clear()
        assert len(cov.sensors) == 1   # internal list unaffected

    def test_sensors_property_contains_sensor(self):
        s   = _attached_sensor()
        cov = Coverage(_aoi(), [s])
        assert cov.sensors[0] is s


# ===========================================================================
# Coverage methods — type and shape checks
# ===========================================================================

class TestCoverageMethods:
    """Verify return types and structure; numeric accuracy is tested in the
    existing functional coverage tests."""

    def setup_method(self):
        self.s   = _attached_sensor()
        self.aoi = _aoi()
        self.cov = Coverage(self.aoi, [self.s])

    def test_coverage_fraction_keys(self):
        result = self.cov.coverage_fraction(
            _EPOCH, _T_END, max_step=_STEP,
        )
        for key in ('t', 'fraction', 'cumulative', 'mean_fraction',
                    'final_cumulative'):
            assert key in result

    def test_coverage_fraction_scalar_outputs(self):
        result = self.cov.coverage_fraction(
            _EPOCH, _T_END, max_step=_STEP,
        )
        assert np.isscalar(result['mean_fraction'])
        assert np.isscalar(result['final_cumulative'])
        assert 0.0 <= result['final_cumulative'] <= 1.0

    def test_revisit_time_keys(self):
        result = self.cov.revisit_time(
            _EPOCH, _T_END, max_step=_STEP,
        )
        for key in ('max_revisit', 'mean_revisit', 'global_max', 'global_mean'):
            assert key in result

    def test_pointwise_coverage_visible_shape(self):
        result = self.cov.pointwise_coverage(
            _EPOCH, _T_END, max_step=_STEP,
        )
        assert 'visible' in result
        N = len(result['t'])
        M = len(self.aoi)
        assert result['visible'].shape == (N, M)
        assert result['visible'].dtype == bool

    def test_access_pointwise_returns_list_of_lists(self):
        result = self.cov.access_pointwise(
            _EPOCH, _T_END, max_step=_STEP,
        )
        assert isinstance(result, list)
        assert len(result) == len(self.aoi)
        for item in result:
            assert isinstance(item, list)

    def test_revisit_pointwise_returns_list_of_arrays(self):
        result = self.cov.revisit_pointwise(
            _EPOCH, _T_END, max_step=_STEP,
        )
        assert isinstance(result, list)
        assert len(result) == len(self.aoi)
        for item in result:
            assert isinstance(item, np.ndarray)


# ===========================================================================
# Constraints forwarded correctly
# ===========================================================================

class TestCoverageConstraints:

    def test_el_min_forwarded(self):
        s   = _attached_sensor()
        cov = Coverage(_aoi(), [s], el_min_deg=30.0)
        # High el_min → fewer accesses than el_min=0
        cov_no_el = Coverage(_aoi(), [s], el_min_deg=0.0)
        r_strict  = cov.coverage_fraction(_EPOCH, _T_END, max_step=_STEP)
        r_loose   = cov_no_el.coverage_fraction(_EPOCH, _T_END, max_step=_STEP)
        assert r_strict['final_cumulative'] <= r_loose['final_cumulative']

    def test_sza_max_reduces_coverage(self):
        s    = _attached_sensor()
        cov_day  = Coverage(_aoi(), [s], sza_max_deg=70.0)
        cov_full = Coverage(_aoi(), [s])
        r_day  = cov_day.coverage_fraction(_EPOCH, _T_END, max_step=_STEP)
        r_full = cov_full.coverage_fraction(_EPOCH, _T_END, max_step=_STEP)
        assert r_day['final_cumulative'] <= r_full['final_cumulative']

    def test_independent_sensor_same_result_as_body_sensor(self):
        """An independent nadir AttitudeLaw sensor and a body-z sensor on a nadir
        spacecraft should produce identical coverage."""
        sc_body = _sc()
        s_body  = Sensor(30.0, body_vector=[0, 0, 1])
        sc_body.add_sensor(s_body)

        sc_ind = _sc()
        s_ind  = Sensor(30.0, attitude_law=AttitudeLaw.nadir())
        sc_ind.add_sensor(s_ind)

        aoi = _aoi()
        cov_body = Coverage(aoi, [s_body])
        cov_ind  = Coverage(aoi, [s_ind])

        r_body = cov_body.coverage_fraction(_EPOCH, _T_END, max_step=_STEP)
        r_ind  = cov_ind.coverage_fraction(_EPOCH, _T_END, max_step=_STEP)
        np.testing.assert_allclose(
            r_body['final_cumulative'], r_ind['final_cumulative'], atol=1e-6,
        )
