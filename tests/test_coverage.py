import numpy as np
import pytest

from missiontools.coverage import (sample_aoi, sample_region, sample_shapefile,
                                   sample_geography,
                                   coverage_fraction, revisit_time, pointwise_coverage,
                                   access_pointwise, revisit_pointwise)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Small rectangular polygon around the UK (lat/lon in radians)
# ~49°–59°N, ~8°W–2°E
_UK_POLY = np.radians([
    [49.0, -8.0],
    [59.0, -8.0],
    [59.0,  2.0],
    [49.0,  2.0],
])

# ISS-like orbit — epoch J2000
_J2000 = np.datetime64('2000-01-01T12:00:00', 'us')
_ISS = dict(
    epoch  = _J2000,
    a      = 6_771_000.0,
    e      = 0.0006,
    i      = np.radians(51.6),
    arg_p  = np.radians(30.0),
    raan   = np.radians(120.0),
    ma     = np.radians(0.0),
)

_T_START = _J2000
_T_END   = _J2000 + np.timedelta64(6 * 3600, 's')   # 6-hour window
_STEP    = np.timedelta64(60, 's')


# ===========================================================================
# sample_aoi
# ===========================================================================

class TestSampleAoi:

    def test_returns_two_arrays(self):
        lat, lon = sample_aoi(_UK_POLY, 50)
        assert isinstance(lat, np.ndarray)
        assert isinstance(lon, np.ndarray)
        assert lat.shape == lon.shape

    def test_approximate_count(self):
        """Returned count should be within 20% of the requested n."""
        n = 200
        lat, lon = sample_aoi(_UK_POLY, n)
        assert 0.8 * n <= len(lat) <= 1.2 * n

    def test_all_points_inside_polygon(self):
        """Every returned point must lie inside the polygon."""
        from matplotlib.path import Path
        lat, lon = sample_aoi(_UK_POLY, 100)
        path = Path(_UK_POLY[:, ::-1])   # (lon, lat)
        inside = path.contains_points(np.column_stack([lon, lat]))
        assert inside.all(), f"{(~inside).sum()} point(s) outside polygon"

    def test_latitudes_in_range(self):
        lat, lon = sample_aoi(_UK_POLY, 50)
        lo, hi = _UK_POLY[:, 0].min(), _UK_POLY[:, 0].max()
        assert (lat >= lo).all() and (lat <= hi).all()

    def test_n_equals_1(self):
        lat, lon = sample_aoi(_UK_POLY, 1)
        assert len(lat) == 1
        assert len(lon) == 1

    def test_invalid_polygon_raises(self):
        with pytest.raises(ValueError, match="polygon"):
            sample_aoi(np.zeros((4, 3)), 10)

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="n must be at least 1"):
            sample_aoi(_UK_POLY, 0)


# ===========================================================================
# coverage_fraction
# ===========================================================================

def _sample_uk(n=30):
    return sample_aoi(_UK_POLY, n)


class TestCoverageFraction:

    def test_output_keys(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        for key in ('t', 'fraction', 'cumulative', 'mean_fraction',
                    'final_cumulative'):
            assert key in result

    def test_array_shapes(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        N = len(result['t'])
        assert N > 0
        assert result['fraction'].shape  == (N,)
        assert result['cumulative'].shape == (N,)

    def test_fraction_in_unit_interval(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        assert (result['fraction']  >= 0.0).all()
        assert (result['fraction']  <= 1.0).all()
        assert (result['cumulative'] >= 0.0).all()
        assert (result['cumulative'] <= 1.0).all()

    def test_cumulative_non_decreasing(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        diffs = np.diff(result['cumulative'])
        assert (diffs >= -1e-6).all(), "cumulative coverage must not decrease"

    def test_mean_fraction_consistent(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        assert abs(result['mean_fraction'] - result['fraction'].mean()) < 1e-4

    def test_final_cumulative_equals_last_element(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        assert abs(result['final_cumulative'] - result['cumulative'][-1]) < 1e-6

    def test_timestamps_are_datetime64(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   max_step=_STEP)
        assert result['t'].dtype == np.dtype('datetime64[us]')
        assert result['t'][0]  == _T_START
        assert result['t'][-1] == _T_END

    def test_empty_window_returns_empty(self):
        lat, lon = _sample_uk()
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_START,
                                   max_step=_STEP)
        assert len(result['t']) == 0

    def test_batching_consistent(self):
        """Result must be identical regardless of batch_size."""
        lat, lon = _sample_uk(20)
        kwargs   = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                        max_step=_STEP)
        r_small = coverage_fraction(lat, lon, batch_size=5,    **kwargs)
        r_large = coverage_fraction(lat, lon, batch_size=5000, **kwargs)
        np.testing.assert_array_equal(r_small['fraction'],  r_large['fraction'])
        np.testing.assert_array_equal(r_small['cumulative'], r_large['cumulative'])

    def test_j2_propagator_runs(self):
        lat, lon = _sample_uk(10)
        result = coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                                   propagator_type='j2', max_step=_STEP)
        assert isinstance(result['mean_fraction'], float)


# ===========================================================================
# revisit_time
# ===========================================================================

class TestRevisitTime:

    def test_output_keys(self):
        lat, lon = _sample_uk()
        result = revisit_time(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP)
        for key in ('max_revisit', 'mean_revisit', 'global_max', 'global_mean'):
            assert key in result

    def test_shapes(self):
        lat, lon = _sample_uk(20)
        result = revisit_time(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP)
        assert result['max_revisit'].shape  == (20,)
        assert result['mean_revisit'].shape == (20,)

    def test_max_ge_mean(self):
        """Per-point max revisit must be ≥ mean revisit (ignoring NaN)."""
        lat, lon = _sample_uk()
        result = revisit_time(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP)
        mx = result['max_revisit']
        mn = result['mean_revisit']
        valid = ~(np.isnan(mx) | np.isnan(mn))
        assert (mx[valid] >= mn[valid] - 1e-6).all()

    def test_positive_revisit_times(self):
        lat, lon = _sample_uk()
        result = revisit_time(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP)
        for arr in (result['max_revisit'], result['mean_revisit']):
            valid = arr[~np.isnan(arr)]
            assert (valid > 0).all()

    def test_global_max_ge_all_per_point(self):
        lat, lon = _sample_uk()
        result = revisit_time(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP)
        if not np.isnan(result['global_max']):
            assert result['global_max'] >= np.nanmax(result['max_revisit']) - 1e-6

    def test_empty_window_returns_nan(self):
        lat, lon = _sample_uk(5)
        result = revisit_time(lat, lon, _ISS, _T_START, _T_START,
                              max_step=_STEP)
        assert np.isnan(result['global_max'])
        assert np.isnan(result['global_mean'])

    def test_batching_consistent(self):
        """Revisit statistics must be identical regardless of batch_size."""
        lat, lon = _sample_uk(15)
        kwargs   = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                        max_step=_STEP)
        r_small = revisit_time(lat, lon, batch_size=5,    **kwargs)
        r_large = revisit_time(lat, lon, batch_size=5000, **kwargs)
        np.testing.assert_array_equal(r_small['max_revisit'],  r_large['max_revisit'])
        np.testing.assert_array_equal(r_small['mean_revisit'], r_large['mean_revisit'])


# ===========================================================================
# sample_region
# ===========================================================================

class TestSampleRegion:

    # --- point count scales with area ---

    def test_global_returns_points(self):
        """All-None call must return a non-empty global sample."""
        lat, lon = sample_region()
        assert len(lat) > 0
        assert len(lat) == len(lon)

    def test_denser_density_gives_more_points(self):
        """Halving point_density should roughly double the point count."""
        lat_a, _ = sample_region(point_density=2e11)
        lat_b, _ = sample_region(point_density=1e11)
        assert len(lat_b) > len(lat_a)

    # --- latitude bounds ---

    def test_lat_band_respects_bounds(self):
        """All returned latitudes must lie within [lat_min, lat_max]."""
        lo, hi = np.radians(30.0), np.radians(60.0)
        lat, _ = sample_region(lat_min=lo, lat_max=hi, point_density=1e11)
        assert (lat >= lo).all() and (lat <= hi).all()

    def test_lat_min_none_reaches_south_pole(self):
        """lat_min=None must allow points below −60°."""
        lat, _ = sample_region(lat_max=np.radians(0.0), point_density=5e12)
        assert lat.min() < np.radians(-60.0)

    def test_lat_max_none_reaches_north_pole(self):
        """lat_max=None must allow points above +60°."""
        lat, _ = sample_region(lat_min=np.radians(0.0), point_density=5e12)
        assert lat.max() > np.radians(60.0)

    # --- longitude bounds ---

    def test_lon_band_respects_bounds(self):
        """All returned longitudes must lie within [lon_min, lon_max]."""
        lo, hi = np.radians(-10.0), np.radians(40.0)
        _, lon = sample_region(lon_min=lo, lon_max=hi, point_density=1e11)
        assert (lon >= lo).all() and (lon <= hi).all()

    def test_antimeridian_no_points_in_gap(self):
        """For a crossing region (lon_min > lon_max) no point should fall
        in the excluded middle band."""
        lo, hi = np.radians(160.0), np.radians(-160.0)   # 320° span
        _, lon = sample_region(lon_min=lo, lon_max=hi, point_density=2e11)
        # Gap is (hi, lo) = (-160°, 160°); no point should be in there
        in_gap = (lon > hi) & (lon < lo)
        assert not in_gap.any()

    def test_antimeridian_has_points_on_both_sides(self):
        """A crossing region should contain points east and west of ±180°."""
        lo, hi = np.radians(150.0), np.radians(-150.0)
        _, lon = sample_region(lon_min=lo, lon_max=hi, point_density=5e11)
        assert (lon >= lo).any(), "no points east of antimeridian"
        assert (lon <= hi).any(), "no points west of antimeridian"

    # --- all-None shorthand ---

    def test_all_none_is_global(self):
        """sample_region() with no arguments should cover all latitudes."""
        lat, lon = sample_region(point_density=5e12)
        assert lat.min() < np.radians(-60.0)
        assert lat.max() > np.radians(60.0)
        # Longitudes should span close to full circle
        assert lon.max() - lon.min() > np.radians(300.0)

    # --- validation ---

    def test_mismatched_lon_raises(self):
        with pytest.raises(ValueError, match="lon_min and lon_max"):
            sample_region(lon_min=np.radians(0.0))

    def test_mismatched_lon_raises_other_direction(self):
        with pytest.raises(ValueError, match="lon_min and lon_max"):
            sample_region(lon_max=np.radians(0.0))

    def test_inverted_lat_raises(self):
        with pytest.raises(ValueError, match="lat_min"):
            sample_region(lat_min=np.radians(60.0), lat_max=np.radians(30.0))

    def test_equal_lat_raises(self):
        with pytest.raises(ValueError, match="lat_min"):
            sample_region(lat_min=np.radians(45.0), lat_max=np.radians(45.0))

    def test_nonpositive_density_raises(self):
        with pytest.raises(ValueError, match="point_density"):
            sample_region(point_density=0.0)

    def test_nonpositive_density_negative_raises(self):
        with pytest.raises(ValueError, match="point_density"):
            sample_region(point_density=-1e10)


# ===========================================================================
# FOV cone constraint
# ===========================================================================

# Nadir pointing in LVLH = -R̂ direction (negative radial)
_NADIR_LVLH = np.array([-1.0, 0.0, 0.0])


class TestCoverageFractionFov:

    def test_wide_nadir_fov_matches_no_fov(self):
        """90° nadir half-angle covers the whole forward hemisphere — should
        equal or nearly equal the horizon-only result."""
        lat, lon = _sample_uk(20)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_base = coverage_fraction(lat, lon, **kw)
        r_fov  = coverage_fraction(lat, lon, **kw,
                                   fov_pointing_lvlh=_NADIR_LVLH,
                                   fov_half_angle=np.radians(90.0))
        np.testing.assert_array_equal(r_base['fraction'], r_fov['fraction'])

    def test_narrow_fov_reduces_coverage(self):
        """A 20° nadir FOV must produce ≤ coverage than no FOV constraint."""
        lat, lon = _sample_uk(20)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_base = coverage_fraction(lat, lon, **kw)
        r_fov  = coverage_fraction(lat, lon, **kw,
                                   fov_pointing_lvlh=_NADIR_LVLH,
                                   fov_half_angle=np.radians(20.0))
        assert r_fov['mean_fraction'] <= r_base['mean_fraction'] + 1e-6
        assert r_fov['final_cumulative'] <= r_base['final_cumulative'] + 1e-6

    def test_pointing_unnormalised_matches_normalised(self):
        """Passing a non-unit pointing vector must give the same result as a
        pre-normalised one."""
        lat, lon = _sample_uk(15)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP, fov_half_angle=np.radians(30.0))
        r_norm   = coverage_fraction(lat, lon, **kw,
                                     fov_pointing_lvlh=_NADIR_LVLH)
        r_scaled = coverage_fraction(lat, lon, **kw,
                                     fov_pointing_lvlh=_NADIR_LVLH * 5.0)
        np.testing.assert_array_equal(r_norm['fraction'], r_scaled['fraction'])

    def test_fov_missing_angle_raises(self):
        """Providing only fov_pointing_lvlh must raise ValueError."""
        lat, lon = _sample_uk(5)
        with pytest.raises(ValueError, match="fov_pointing_lvlh"):
            coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP,
                              fov_pointing_lvlh=_NADIR_LVLH)

    def test_fov_missing_pointing_raises(self):
        """Providing only fov_half_angle must raise ValueError."""
        lat, lon = _sample_uk(5)
        with pytest.raises(ValueError, match="fov_pointing_lvlh"):
            coverage_fraction(lat, lon, _ISS, _T_START, _T_END,
                              max_step=_STEP,
                              fov_half_angle=np.radians(30.0))


class TestRevisitTimeFov:

    def test_fov_reduces_or_equals_revisit(self):
        """With a narrow FOV each point is visited less often, so global_mean
        revisit time must be ≥ the unconstrained case (or NaN if never seen)."""
        lat, lon = _sample_uk(15)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_base = revisit_time(lat, lon, **kw)
        r_fov  = revisit_time(lat, lon, **kw,
                              fov_pointing_lvlh=_NADIR_LVLH,
                              fov_half_angle=np.radians(20.0))
        base_mean = r_base['global_mean']
        fov_mean  = r_fov['global_mean']
        if not np.isnan(fov_mean) and not np.isnan(base_mean):
            assert fov_mean >= base_mean - 1e-6

    def test_revisit_fov_missing_angle_raises(self):
        """Providing only fov_pointing_lvlh must raise ValueError."""
        lat, lon = _sample_uk(5)
        with pytest.raises(ValueError, match="fov_pointing_lvlh"):
            revisit_time(lat, lon, _ISS, _T_START, _T_END,
                         max_step=_STEP,
                         fov_pointing_lvlh=_NADIR_LVLH)


# ===========================================================================
# Solar zenith angle constraint
# ===========================================================================

_SZA_90 = np.radians(90.0)   # horizon — splits day/night exactly


class TestCoverageFractionSza:

    def test_sza_daytime_reduces_or_equals_coverage(self):
        """Restricting to daytime (SZA ≤ 90°) gives ≤ fraction than unconstrained."""
        lat, lon = _sample_uk(20)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_base = coverage_fraction(lat, lon, **kw)
        r_day  = coverage_fraction(lat, lon, **kw, sza_max=_SZA_90)
        assert r_day['mean_fraction'] <= r_base['mean_fraction'] + 1e-6

    def test_sza_nighttime_reduces_or_equals_coverage(self):
        """Restricting to nighttime (SZA ≥ 90°) gives ≤ fraction than unconstrained."""
        lat, lon = _sample_uk(20)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_base  = coverage_fraction(lat, lon, **kw)
        r_night = coverage_fraction(lat, lon, **kw, sza_min=_SZA_90)
        assert r_night['mean_fraction'] <= r_base['mean_fraction'] + 1e-6

    def test_sza_day_and_night_partition_unconstrained(self):
        """Per-timestep day + night fractions must sum to the unconstrained fraction."""
        lat, lon = _sample_uk(20)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_base  = coverage_fraction(lat, lon, **kw)
        r_day   = coverage_fraction(lat, lon, **kw, sza_max=_SZA_90)
        r_night = coverage_fraction(lat, lon, **kw, sza_min=_SZA_90)
        combined = r_day['fraction'] + r_night['fraction']
        np.testing.assert_allclose(combined, r_base['fraction'], atol=1e-6)

    def test_sza_full_range_matches_no_constraint(self):
        """sza_max=π covers all illumination angles — identical to no constraint."""
        lat, lon = _sample_uk(20)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_base = coverage_fraction(lat, lon, **kw)
        r_all  = coverage_fraction(lat, lon, **kw, sza_max=np.radians(180.0))
        np.testing.assert_array_equal(r_base['fraction'], r_all['fraction'])

    def test_sza_no_overlap_day_night(self):
        """A timestep cannot be both daytime and nighttime for the same point."""
        lat, lon = _sample_uk(20)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_day   = coverage_fraction(lat, lon, **kw, sza_max=_SZA_90)
        r_night = coverage_fraction(lat, lon, **kw, sza_min=_SZA_90)
        # At each timestep a point is either day or night, never both
        overlap = r_day['fraction'] * r_night['fraction']
        # overlap > 0 only if same point is simultaneously day AND night, impossible
        # (points exactly at the terminator can round to both; allow small tolerance)
        assert np.all(overlap < 1e-5)


class TestRevisitTimeSza:

    def test_sza_daytime_revisit_ge_unconstrained(self):
        """Daytime-only revisit time must be ≥ unconstrained (fewer valid passes)."""
        lat, lon = _sample_uk(15)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                  max_step=_STEP)
        r_base = revisit_time(lat, lon, **kw)
        r_day  = revisit_time(lat, lon, **kw, sza_max=_SZA_90)
        base_mean = r_base['global_mean']
        day_mean  = r_day['global_mean']
        if not np.isnan(day_mean) and not np.isnan(base_mean):
            assert day_mean >= base_mean - 1e-6


# ===========================================================================
# pointwise_coverage
# ===========================================================================

_N_PW_PTS = 20   # small M for speed

def _kw_pw(**extra):
    """Common kwargs for pointwise_coverage tests."""
    return dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                max_step=_STEP, **extra)


class TestPointwiseCoverage:

    def test_shape(self):
        """visible must have shape (N, M) and t shape (N,)."""
        lat, lon = _sample_uk(_N_PW_PTS)
        result = pointwise_coverage(lat, lon, **_kw_pw())
        N = len(result['t'])
        M = len(lat)
        assert result['visible'].shape == (N, M)
        assert result['visible'].dtype == np.bool_

    def test_lat_lon_alt_passthrough(self):
        """Returned lat/lon/alt must match the inputs exactly."""
        lat, lon = _sample_uk(_N_PW_PTS)
        result = pointwise_coverage(lat, lon, **_kw_pw(alt=500.0))
        np.testing.assert_array_equal(result['lat'], lat)
        np.testing.assert_array_equal(result['lon'], lon)
        assert result['alt'] == 500.0

    def test_consistent_with_coverage_fraction(self):
        """visible.mean(axis=1) must equal coverage_fraction 'fraction' output."""
        lat, lon = _sample_uk(_N_PW_PTS)
        kw = _kw_pw()
        pw = pointwise_coverage(lat, lon, **kw)
        cf = coverage_fraction(lat, lon, **kw)
        # Both use the same propagation path — agreement should be exact
        inst = pw['visible'].mean(axis=1).astype(np.float32)
        np.testing.assert_allclose(inst, cf['fraction'], atol=1e-6)

    def test_with_fov(self):
        """FOV constraint: every constrained point is also in unconstrained set."""
        lat, lon = _sample_uk(_N_PW_PTS)
        nadir = np.array([-1.0, 0.0, 0.0])
        kw = _kw_pw()
        base = pointwise_coverage(lat, lon, **kw)
        fov  = pointwise_coverage(lat, lon, **kw,
                                  fov_pointing_lvlh=nadir,
                                  fov_half_angle=np.radians(30.0))
        # FOV-constrained visibility is a subset of unconstrained
        assert (fov['visible'] & ~base['visible']).sum() == 0

    def test_with_sza(self):
        """SZA constraint: daytime-only visible ≤ unconstrained."""
        lat, lon = _sample_uk(_N_PW_PTS)
        kw = _kw_pw()
        base = pointwise_coverage(lat, lon, **kw)
        day  = pointwise_coverage(lat, lon, **kw, sza_max=np.pi / 2)
        assert (day['visible'] & ~base['visible']).sum() == 0

    def test_empty_window(self):
        """t_start == t_end returns empty arrays without raising."""
        lat, lon = _sample_uk(5)
        result = pointwise_coverage(lat, lon,
                                    keplerian_params=_ISS,
                                    t_start=_T_START, t_end=_T_START,
                                    max_step=_STEP)
        assert len(result['t']) == 0
        assert result['visible'].shape == (0, 5)


# ===========================================================================
# sample_shapefile
# ===========================================================================

def _make_shapefile(tmp_path, rings_lon_lat_deg, name='test'):
    """Write a single-feature Polygon shapefile; return path to .shp."""
    import shapefile as _pyshp
    path = str(tmp_path / f'{name}.shp')
    w = _pyshp.Writer(path, shapeType=5)
    w.field('name', 'C')

    def _area2(ring):
        n = len(ring)
        return sum(ring[i][0] * ring[(i + 1) % n][1] -
                   ring[(i + 1) % n][0] * ring[i][1]
                   for i in range(n))

    # ESRI convention: exterior rings are CW (negative signed area),
    # hole rings are CCW (positive signed area).
    fixed = []
    for i, ring in enumerate(rings_lon_lat_deg):
        pts = list(ring)
        a2 = _area2(pts)
        if i == 0:      # exterior: force CW
            if a2 > 0:
                pts = pts[::-1]
        else:           # hole: force CCW
            if a2 < 0:
                pts = pts[::-1]
        fixed.append(pts)

    w.poly(fixed)
    w.record(name)
    w.close()
    return path


def _make_two_feature_shapefile(tmp_path, rings_a, rings_b):
    """Write a two-feature Polygon shapefile; return path to .shp."""
    import shapefile as _pyshp
    path = str(tmp_path / 'two.shp')
    w = _pyshp.Writer(path, shapeType=5)
    w.field('name', 'C')
    w.poly(rings_a); w.record('A')
    w.poly(rings_b); w.record('B')
    w.close()
    return path


# ===========================================================================
# access_pointwise
# ===========================================================================

def _kw_ap(**extra):
    return dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_END,
                max_step=_STEP, **extra)


class TestAccessPointwise:

    def test_outer_len_equals_point_count(self):
        lat, lon = _sample_uk(_N_PW_PTS)
        result = access_pointwise(lat, lon, **_kw_ap())
        assert len(result) == _N_PW_PTS

    def test_inner_type(self):
        lat, lon = _sample_uk(_N_PW_PTS)
        result = access_pointwise(lat, lon, **_kw_ap())
        assert all(isinstance(result[m], list) for m in range(_N_PW_PTS))

    def test_interval_type(self):
        lat, lon = _sample_uk(_N_PW_PTS)
        result = access_pointwise(lat, lon, **_kw_ap())
        for m in range(_N_PW_PTS):
            for a, l in result[m]:
                assert np.issubdtype(type(a), np.datetime64)
                assert np.issubdtype(type(l), np.datetime64)

    def test_aos_before_los(self):
        lat, lon = _sample_uk(_N_PW_PTS)
        result = access_pointwise(lat, lon, **_kw_ap())
        for ivals in result:
            for a, l in ivals:
                assert a < l

    def test_intervals_sorted_per_point(self):
        lat, lon = _sample_uk(_N_PW_PTS)
        result = access_pointwise(lat, lon, **_kw_ap())
        for ivals in result:
            aos = [a for a, _ in ivals]
            assert aos == sorted(aos)

    def test_consistent_with_pointwise_coverage(self):
        """Every True cell in pointwise_coverage must fall inside an interval."""
        lat, lon = _sample_uk(_N_PW_PTS)
        kw = _kw_ap()
        pw  = pointwise_coverage(lat, lon, **kw)
        aps = access_pointwise(lat, lon, **kw)
        t_arr = pw['t']
        for m in range(_N_PW_PTS):
            for ti, t in enumerate(t_arr):
                if pw['visible'][ti, m]:
                    covered = any(a <= t <= l for a, l in aps[m])
                    assert covered, (
                        f"point {m}: visible at t={t} but not in any interval"
                    )

    def test_empty_window(self):
        lat, lon = _sample_uk(5)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_START, max_step=_STEP)
        result = access_pointwise(lat, lon, **kw)
        assert all(result[m] == [] for m in range(5))

    def test_fov_gives_subset(self):
        """Adding a tight FOV constraint reduces or preserves interval count."""
        lat, lon = _sample_uk(_N_PW_PTS)
        base = access_pointwise(lat, lon, **_kw_ap())
        fov  = access_pointwise(lat, lon, **_kw_ap(
            fov_pointing_lvlh=np.array([0.0, 0.0, 1.0]),
            fov_half_angle=np.radians(20.0),
        ))
        total_base = sum(len(iv) for iv in base)
        total_fov  = sum(len(iv) for iv in fov)
        assert total_fov <= total_base


# ===========================================================================
# revisit_pointwise
# ===========================================================================

class TestRevisitPointwise:

    def test_outer_len_equals_point_count(self):
        lat, lon = _sample_uk(_N_PW_PTS)
        result = revisit_pointwise(lat, lon, **_kw_ap())
        assert len(result) == _N_PW_PTS

    def test_entries_are_ndarray_of_timedelta(self):
        lat, lon = _sample_uk(_N_PW_PTS)
        result = revisit_pointwise(lat, lon, **_kw_ap())
        for arr in result:
            assert isinstance(arr, np.ndarray)
            assert np.issubdtype(arr.dtype, np.timedelta64)

    def test_gaps_positive(self):
        lat, lon = _sample_uk(_N_PW_PTS)
        result = revisit_pointwise(lat, lon, **_kw_ap())
        for arr in result:
            if len(arr) > 0:
                assert (arr > np.timedelta64(0, 'us')).all()

    def test_consistent_with_revisit_time_max(self):
        """Max gap in revisit_pointwise[m] must equal revisit_time max_revisit[m]."""
        lat, lon = _sample_uk(_N_PW_PTS)
        kw = _kw_ap()
        rp  = revisit_pointwise(lat, lon, **kw)
        rt  = revisit_time(lat, lon, **kw)
        for m in range(_N_PW_PTS):
            if len(rp[m]) > 0:
                max_s = float(rp[m].max() / np.timedelta64(1, 'us')) / 1e6
                np.testing.assert_allclose(max_s, rt['max_revisit'][m], rtol=1e-9)
            else:
                assert np.isnan(rt['max_revisit'][m])

    def test_consistent_with_revisit_time_mean(self):
        lat, lon = _sample_uk(_N_PW_PTS)
        kw = _kw_ap()
        rp  = revisit_pointwise(lat, lon, **kw)
        rt  = revisit_time(lat, lon, **kw)
        for m in range(_N_PW_PTS):
            if len(rp[m]) > 0:
                mean_s = float(rp[m].mean() / np.timedelta64(1, 'us')) / 1e6
                np.testing.assert_allclose(mean_s, rt['mean_revisit'][m], rtol=1e-9)

    def test_fewer_than_two_intervals_gives_empty(self):
        """Points with ≤ 1 access window should have an empty gap array."""
        lat, lon = _sample_uk(_N_PW_PTS)
        aps = access_pointwise(lat, lon, **_kw_ap())
        rp  = revisit_pointwise(lat, lon, **_kw_ap())
        for m in range(_N_PW_PTS):
            if len(aps[m]) < 2:
                assert len(rp[m]) == 0

    def test_empty_window(self):
        lat, lon = _sample_uk(5)
        kw = dict(keplerian_params=_ISS, t_start=_T_START, t_end=_T_START, max_step=_STEP)
        result = revisit_pointwise(lat, lon, **kw)
        assert all(len(arr) == 0 for arr in result)


# ===========================================================================
# sample_shapefile
# ===========================================================================

# A roughly 10° × 10° box centred on 0°N, 10°E (in degrees, lon before lat)
_BOX_10E = [[(5.0, -5.0), (15.0, -5.0), (15.0, 5.0), (5.0, 5.0), (5.0, -5.0)]]

# A roughly 10° × 10° box centred on 30°N, 30°E
_BOX_30E = [[(25.0, 25.0), (35.0, 25.0), (35.0, 35.0), (25.0, 35.0), (25.0, 25.0)]]

# A small box centred exactly on the antimeridian: lons 175° → 185° (= −175°)
# Represented with extended longitudes so the ring doesn't jump.
_BOX_AM  = [[(175.0, -5.0), (185.0, -5.0), (185.0, 5.0), (175.0, 5.0), (175.0, -5.0)]]


class TestSampleShapefile:

    def test_returns_two_arrays(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        lat, lon = sample_shapefile(path, point_density=5e12)
        assert isinstance(lat, np.ndarray)
        assert isinstance(lon, np.ndarray)
        assert lat.shape == lon.shape
        assert len(lat) > 0

    def test_all_inside_polygon(self, tmp_path):
        """Every returned point must lie inside the shapefile geometry."""
        import shapely
        from shapely.geometry import Polygon
        path = _make_shapefile(tmp_path, _BOX_10E)
        lat, lon = sample_shapefile(path, point_density=5e12)
        geom = Polygon(_BOX_10E[0])
        inside = shapely.contains_xy(geom, np.degrees(lon), np.degrees(lat))
        assert inside.all(), f"{(~inside).sum()} point(s) outside polygon"

    def test_denser_density_gives_more_points(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        _, lon_coarse = sample_shapefile(path, point_density=2e11)
        _, lon_dense  = sample_shapefile(path, point_density=5e10)
        assert len(lon_dense) > len(lon_coarse)

    def test_feature_index_single(self, tmp_path):
        """feature_index selects one shape; lat range is within that feature only."""
        path = _make_two_feature_shapefile(tmp_path, _BOX_10E, _BOX_30E)
        # feature 0 is _BOX_10E (lat -5°…5°), feature 1 is _BOX_30E (lat 25°…35°)
        lat_f0, _ = sample_shapefile(path, feature_index=0, point_density=1e10)
        assert len(lat_f0) > 0
        # All points must lie in the latitude range of feature 0 (±5°)
        assert (np.degrees(lat_f0) >= -5.5).all()
        assert (np.degrees(lat_f0) <=  5.5).all()

    def test_polygon_with_hole(self, tmp_path):
        """Points must not fall inside the hole of a donut polygon."""
        import shapely
        from shapely.geometry import Polygon
        # Outer ring 5°×5°; inner hole 2°×2°
        outer = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]
        hole  = [(1.5, 1.5), (3.5, 1.5), (3.5, 3.5), (1.5, 3.5), (1.5, 1.5)]
        path  = _make_shapefile(tmp_path, [outer, hole])
        lat, lon = sample_shapefile(path, point_density=1e11)
        hole_geom = Polygon(hole)
        in_hole = shapely.contains_xy(hole_geom, np.degrees(lon), np.degrees(lat))
        assert not in_hole.any(), f"{in_hole.sum()} point(s) inside hole"

    def test_antimeridian_polygon(self, tmp_path):
        """Points from an antimeridian-crossing box must be near ±180°."""
        path = _make_shapefile(tmp_path, _BOX_AM)
        lat, lon = sample_shapefile(path, point_density=5e12)
        assert len(lat) > 0
        # All returned longitudes should be > 175° or < -175°
        lon_deg = np.degrees(lon)
        assert ((lon_deg > 175.0) | (lon_deg < -175.0)).all(), \
            f"Points not near antimeridian: {lon_deg}"

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(Exception):   # pyshp raises ShapefileException (OSError subclass)
            sample_shapefile(str(tmp_path / 'missing.shp'))

    def test_nonpolygon_shapefile_raises(self, tmp_path):
        """A point or polyline shapefile must raise ValueError."""
        import shapefile as _pyshp
        path = str(tmp_path / 'pts.shp')
        w = _pyshp.Writer(path, shapeType=1)   # POINT
        w.field('id', 'N')
        w.point(10.0, 20.0)
        w.record(1)
        w.close()
        with pytest.raises(ValueError, match="No polygon features"):
            sample_shapefile(path)

    def test_nonpositive_density_raises(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        with pytest.raises(ValueError, match="point_density"):
            sample_shapefile(path, point_density=0)


# ===========================================================================
# sample_geography
# ===========================================================================

class TestSampleGeography:
    """Tests use the bundled Natural Earth 50 m dataset."""

    def test_returns_three_tuple(self):
        lat, lon, geom = sample_geography('Canada')
        assert isinstance(lat, np.ndarray)
        assert isinstance(lon, np.ndarray)
        assert lat.shape == lon.shape
        assert lat.ndim == 1
        assert geom is not None

    def test_geometry_is_shapely_polygon(self):
        import shapely
        _, _, geom = sample_geography('Canada')
        assert isinstance(geom, (shapely.geometry.Polygon,
                                 shapely.geometry.MultiPolygon))

    def test_lat_lon_in_radians(self):
        lat, lon, _ = sample_geography('Canada')
        assert np.all(lat >= -np.pi / 2) and np.all(lat <= np.pi / 2)
        # lon may be extended past ±π for antimeridian crossers; Canada
        # does not cross the AM so check standard range
        assert np.all(lon >= -np.pi) and np.all(lon <= np.pi)

    # ── Name lookup ──────────────────────────────────────────────────────────

    def test_name_lookup_country(self):
        lat, lon, geom = sample_geography('Canada')
        assert len(lat) > 0

    def test_name_lookup_case_insensitive(self):
        lat1, _, _ = sample_geography('Canada')
        lat2, _, _ = sample_geography('canada')
        assert len(lat1) == len(lat2)

    # ── ISO A2 ───────────────────────────────────────────────────────────────

    def test_iso_a2_same_feature_count_as_name(self):
        lat_name, _, _ = sample_geography('Canada')
        lat_iso,  _, _ = sample_geography('CA')
        # Same feature → same point count (density-based, so equal)
        assert len(lat_name) == len(lat_iso)

    # ── ISO A3 ───────────────────────────────────────────────────────────────

    def test_iso_a3_same_as_iso_a2(self):
        lat_a2, _, _ = sample_geography('CA')
        lat_a3, _, _ = sample_geography('CAN')
        assert len(lat_a2) == len(lat_a3)

    # ── Slash pattern ────────────────────────────────────────────────────────

    def test_slash_pattern_nonempty(self):
        lat_qc, lon_qc, _ = sample_geography('Canada/Quebec')
        assert len(lat_qc) > 0

    def test_slash_pattern_points_in_quebec_bbox(self):
        lat, lon, _ = sample_geography('Canada/Quebec')
        lat_deg = np.degrees(lat)
        lon_deg = np.degrees(lon)
        # Quebec rough bounding box
        assert np.all(lat_deg >= 44.0) and np.all(lat_deg <= 63.0)
        assert np.all(lon_deg >= -80.0) and np.all(lon_deg <= -56.0)

    def test_slash_pattern_case_insensitive(self):
        lat1, _, _ = sample_geography('Canada/Quebec')
        lat2, _, _ = sample_geography('canada/quebec')
        assert len(lat1) == len(lat2)

    # ── ISO 3166-2 ───────────────────────────────────────────────────────────

    def test_iso_3166_2_same_as_slash(self):
        lat_slash, _, _ = sample_geography('Canada/Quebec')
        lat_iso,   _, _ = sample_geography('CA-QC')
        assert len(lat_slash) == len(lat_iso)

    # ── Error handling ───────────────────────────────────────────────────────

    def test_unknown_geography_raises(self):
        with pytest.raises(ValueError, match="Geography not found"):
            sample_geography('Narnia')

    def test_invalid_subdivision_raises(self):
        with pytest.raises(ValueError):
            sample_geography('Canada/Narnia')

    def test_nonpositive_density_raises(self):
        with pytest.raises(ValueError, match="point_density"):
            sample_geography('Canada', point_density=0)

    # ── Regression guard: sample_shapefile refactor unchanged ────────────────

    def test_sample_shapefile_still_works(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        lat, lon = sample_shapefile(path, point_density=5e12)
        assert len(lat) > 0
        assert lat.shape == lon.shape
