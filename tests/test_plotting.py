"""
Tests for missiontools.plotting.

The Cartopy-backed tests are skipped automatically if cartopy is not installed.
The helper-function tests (_ecef_to_latlon, _split_antimeridian, _set_extent)
do not require Cartopy and run unconditionally.
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers for creating test fixtures
# ---------------------------------------------------------------------------

def _make_spacecraft():
    from missiontools import Spacecraft
    return Spacecraft(
        a=6_771_000., e=0., i=np.radians(51.6),
        raan=0., arg_p=0., ma=0.,
        epoch=np.datetime64('2025-01-01', 'us'),
    )


def _make_aoi():
    from missiontools import AoI
    return AoI.from_region(-30, 30, -60, 60)


# ===========================================================================
# TestEcefToLatlon — pure maths, no Cartopy required
# ===========================================================================

class TestEcefToLatlon:
    from missiontools.plotting.ground_track import _ecef_to_latlon

    def test_plus_x_axis(self):
        from missiontools.plotting.ground_track import _ecef_to_latlon
        r = np.array([[1.0, 0.0, 0.0]])
        lat, lon = _ecef_to_latlon(r)
        assert np.isclose(lat[0], 0.0, atol=1e-10)
        assert np.isclose(lon[0], 0.0, atol=1e-10)

    def test_plus_y_axis(self):
        from missiontools.plotting.ground_track import _ecef_to_latlon
        r = np.array([[0.0, 1.0, 0.0]])
        lat, lon = _ecef_to_latlon(r)
        assert np.isclose(lat[0], 0.0, atol=1e-10)
        assert np.isclose(lon[0], 90.0, atol=1e-10)

    def test_plus_z_axis(self):
        from missiontools.plotting.ground_track import _ecef_to_latlon
        r = np.array([[0.0, 0.0, 1.0]])
        lat, lon = _ecef_to_latlon(r)
        assert np.isclose(lat[0], 90.0, atol=1e-10)

    def test_minus_x_axis(self):
        from missiontools.plotting.ground_track import _ecef_to_latlon
        r = np.array([[-1.0, 0.0, 0.0]])
        lat, lon = _ecef_to_latlon(r)
        assert np.isclose(lat[0], 0.0, atol=1e-10)
        assert np.isclose(abs(lon[0]), 180.0, atol=1e-10)

    def test_batch_shape(self):
        from missiontools.plotting.ground_track import _ecef_to_latlon
        r = np.random.randn(50, 3)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        lat, lon = _ecef_to_latlon(r)
        assert lat.shape == (50,)
        assert lon.shape == (50,)
        assert np.all(lat >= -90.) and np.all(lat <= 90.)
        assert np.all(lon >= -180.) and np.all(lon <= 180.)


# ===========================================================================
# TestSplitAntimeridian — pure, no Cartopy required
# ===========================================================================

class TestSplitAntimeridian:

    def test_continuous_track_one_segment(self):
        from missiontools.plotting.ground_track import _split_antimeridian
        lat = np.array([10., 20., 30.])
        lon = np.array([0.,  45.,  90.])
        segs = _split_antimeridian(lat, lon)
        assert len(segs) == 1
        np.testing.assert_array_equal(segs[0][0], lat)
        np.testing.assert_array_equal(segs[0][1], lon)

    def test_antimeridian_crossing_two_segments(self):
        from missiontools.plotting.ground_track import _split_antimeridian
        lat = np.array([0., 0., 0., 0.])
        lon = np.array([170., 175., -175., -170.])
        segs = _split_antimeridian(lat, lon)
        assert len(segs) == 2
        assert len(segs[0][0]) == 2
        assert len(segs[1][0]) == 2

    def test_two_crossings_three_segments(self):
        from missiontools.plotting.ground_track import _split_antimeridian
        lat = np.zeros(6)
        lon = np.array([170., 175., -175., -170., 170., 175.])
        segs = _split_antimeridian(lat, lon)
        assert len(segs) == 3

    def test_single_point(self):
        from missiontools.plotting.ground_track import _split_antimeridian
        lat = np.array([45.])
        lon = np.array([90.])
        segs = _split_antimeridian(lat, lon)
        assert len(segs) == 1


# ===========================================================================
# Cartopy-dependent tests
# ===========================================================================

cartopy = pytest.importorskip('cartopy')


class TestSetExtent:
    """_set_extent computes the bounding box in projected coordinates."""

    def _mock_ax(self, projection=None):
        """Return a mock GeoAxes that records set_extent calls."""
        import cartopy.crs as ccrs

        class MockAx:
            def __init__(self, proj):
                self.projection   = proj
                self.extent_calls = []
                self.crs_calls    = []

            def set_extent(self, extent, crs=None):
                self.extent_calls.append(extent)
                self.crs_calls.append(crs)

        return MockAx(projection or ccrs.PlateCarree())

    def test_padding_symmetric_platecarree(self):
        """For PlateCarree, projected coords == lon/lat so padding is in degrees."""
        from missiontools.plotting._map import _set_extent

        ax  = self._mock_ax()
        lat = np.array([-10., 10.])
        lon = np.array([-20., 20.])
        _set_extent(ax, lat, lon, factor=1.5)

        assert len(ax.extent_calls) == 1
        ext = ax.extent_calls[0]
        # lon range=40°, pad=0.25×40=10; lat range=20°, pad=0.25×20=5
        assert np.isclose(ext[0], -30.0, atol=1e-6)
        assert np.isclose(ext[1],  30.0, atol=1e-6)
        assert np.isclose(ext[2], -15.0, atol=1e-6)
        assert np.isclose(ext[3],  15.0, atol=1e-6)

    def test_extent_crs_matches_projection(self):
        """set_extent is called with the axes' own projection, not PlateCarree."""
        import cartopy.crs as ccrs
        from missiontools.plotting._map import _set_extent

        proj = ccrs.LambertConformal()
        ax   = self._mock_ax(projection=proj)
        lat  = np.linspace(42, 83, 20)
        lon  = np.linspace(-141, -52, 20)
        _set_extent(ax, lat, lon, factor=1.5)

        assert ax.crs_calls[0] is proj

    def test_padding_symmetric_projected(self):
        """Padding is symmetric in projected (metre) units for a conic projection."""
        import cartopy.crs as ccrs
        from missiontools.plotting._map import _set_extent

        proj = ccrs.LambertConformal(central_longitude=-96)
        ax   = self._mock_ax(projection=proj)
        lat  = np.linspace(42, 83, 30)
        lon  = np.linspace(-141, -52, 30)
        _set_extent(ax, lat, lon, factor=1.5)

        ext = ax.extent_calls[0]
        # Transform the same points to verify the extent is sensible
        pts = proj.transform_points(ccrs.PlateCarree(), lon, lat)
        x, y = pts[:, 0], pts[:, 1]
        valid = np.isfinite(x) & np.isfinite(y)
        x_min, x_max = x[valid].min(), x[valid].max()
        y_min, y_max = y[valid].min(), y[valid].max()
        # The extent must contain all data points
        assert ext[0] <= x_min
        assert ext[1] >= x_max
        assert ext[2] <= y_min
        assert ext[3] >= y_max

    def test_minimum_range_guard(self):
        """A single-point AoI still produces a non-zero window."""
        from missiontools.plotting._map import _set_extent

        ax  = self._mock_ax()
        lat = np.array([45., 45.])
        lon = np.array([0.,  0.])
        _set_extent(ax, lat, lon, factor=1.5)

        ext = ax.extent_calls[0]
        assert ext[1] - ext[0] > 0
        assert ext[3] - ext[2] > 0


class TestPlotGroundTrack:

    def test_returns_geoaxes(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.geodesic  # noqa: F401 — ensure cartopy available
        from cartopy.mpl.geoaxes import GeoAxes
        from missiontools.plotting import plot_ground_track

        sc = _make_spacecraft()
        t0 = np.datetime64('2025-01-01', 'us')
        ax = plot_ground_track(sc, t0, t0 + np.timedelta64(5400, 's'))
        assert isinstance(ax, GeoAxes)
        plt.close('all')

    def test_existing_ax_not_replaced(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        from cartopy.mpl.geoaxes import GeoAxes
        from missiontools.plotting import plot_ground_track

        fig, existing_ax = plt.subplots(
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        sc = _make_spacecraft()
        t0 = np.datetime64('2025-01-01', 'us')
        returned_ax = plot_ground_track(
            sc, t0, t0 + np.timedelta64(5400, 's'), ax=existing_ax
        )
        assert returned_ax is existing_ax
        plt.close('all')

    def test_auto_window_does_not_raise(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from missiontools.plotting import plot_ground_track

        sc = _make_spacecraft()
        t0 = np.datetime64('2025-01-01', 'us')
        plot_ground_track(
            sc, t0, t0 + np.timedelta64(5400, 's'), auto_window=True
        )
        plt.close('all')

    def test_custom_projection(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        from missiontools.plotting import plot_ground_track

        sc = _make_spacecraft()
        t0 = np.datetime64('2025-01-01', 'us')
        ax = plot_ground_track(
            sc, t0, t0 + np.timedelta64(5400, 's'),
            projection=ccrs.Mollweide(),
        )
        assert ax is not None
        plt.close('all')

    def test_add_start_marker_false(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from missiontools.plotting import plot_ground_track

        sc = _make_spacecraft()
        t0 = np.datetime64('2025-01-01', 'us')
        ax = plot_ground_track(
            sc, t0, t0 + np.timedelta64(5400, 's'), add_start_marker=False
        )
        assert ax is not None
        plt.close('all')


class TestPlotCoverageMap:

    def test_returns_geoaxes(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from cartopy.mpl.geoaxes import GeoAxes
        from missiontools.plotting import plot_coverage_map

        aoi = _make_aoi()
        values = np.random.rand(len(aoi))
        ax = plot_coverage_map(aoi, values)
        assert isinstance(ax, GeoAxes)
        plt.close('all')

    def test_mismatched_values_raises(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from missiontools.plotting import plot_coverage_map

        aoi = _make_aoi()
        with pytest.raises(ValueError, match=r"len\(values\)"):
            plot_coverage_map(aoi, np.zeros(len(aoi) + 5))
        plt.close('all')

    def test_colorbar_false_does_not_raise(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from missiontools.plotting import plot_coverage_map

        aoi = _make_aoi()
        values = np.random.rand(len(aoi))
        ax = plot_coverage_map(aoi, values, colorbar=False)
        assert ax is not None
        plt.close('all')

    def test_auto_window_does_not_raise(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from missiontools.plotting import plot_coverage_map

        aoi = _make_aoi()
        values = np.random.rand(len(aoi))
        ax = plot_coverage_map(aoi, values, auto_window=True)
        assert ax is not None
        plt.close('all')

    def test_custom_projection(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        from missiontools.plotting import plot_coverage_map

        aoi = _make_aoi()
        values = np.random.rand(len(aoi))
        ax = plot_coverage_map(
            aoi, values,
            projection=ccrs.Mollweide(),
            colorbar=False,
        )
        assert ax is not None
        plt.close('all')

    def test_existing_ax_not_replaced(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        from missiontools.plotting import plot_coverage_map

        _, existing_ax = plt.subplots(
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        aoi = _make_aoi()
        values = np.random.rand(len(aoi))
        returned_ax = plot_coverage_map(aoi, values, ax=existing_ax,
                                        colorbar=False)
        assert returned_ax is existing_ax
        plt.close('all')

    def test_title_and_colorbar_label(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from missiontools.plotting import plot_coverage_map

        aoi = _make_aoi()
        values = np.random.rand(len(aoi))
        ax = plot_coverage_map(
            aoi, values,
            title='Test title',
            colorbar_label='Fraction',
        )
        assert ax.get_title() == 'Test title'
        plt.close('all')
