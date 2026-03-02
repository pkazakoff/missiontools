import numpy as np
import pytest

from missiontools import AoI


# ---------------------------------------------------------------------------
# Helpers / shared data
# ---------------------------------------------------------------------------

_LAT_DEG = np.array([-10.0, 0.0, 10.0])
_LON_DEG = np.array([30.0, 45.0, 60.0])

# Shapefile helper (mirrors test_coverage.py)
def _make_shapefile(tmp_path, rings_lon_lat_deg, name='test'):
    """Write a single-feature Polygon shapefile; return path to .shp."""
    import shapefile as _pyshp  # noqa: PLC0415
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


# A roughly 10° × 10° box centred on 0°N, 10°E (lon before lat)
_BOX_10E = [[(5.0, -5.0), (15.0, -5.0), (15.0, 5.0), (5.0, 5.0), (5.0, -5.0)]]


# ---------------------------------------------------------------------------
# TestAoIDirectConstruct
# ---------------------------------------------------------------------------

class TestAoIDirectConstruct:

    def test_lat_lon_degrees(self):
        aoi = AoI(_LAT_DEG, _LON_DEG)
        np.testing.assert_allclose(aoi.lat, _LAT_DEG)
        np.testing.assert_allclose(aoi.lon, _LON_DEG)

    def test_lat_lon_rad(self):
        aoi = AoI(_LAT_DEG, _LON_DEG)
        np.testing.assert_allclose(aoi.lat_rad, np.radians(_LAT_DEG))
        np.testing.assert_allclose(aoi.lon_rad, np.radians(_LON_DEG))

    def test_internal_storage_is_radians(self):
        aoi = AoI(_LAT_DEG, _LON_DEG)
        np.testing.assert_allclose(aoi._lat, np.radians(_LAT_DEG))
        np.testing.assert_allclose(aoi._lon, np.radians(_LON_DEG))

    def test_len(self):
        aoi = AoI(_LAT_DEG, _LON_DEG)
        assert len(aoi) == 3

    def test_geometry_none(self):
        aoi = AoI(_LAT_DEG, _LON_DEG)
        assert aoi.geometry is None

    def test_shapefile_path_none(self):
        aoi = AoI(_LAT_DEG, _LON_DEG)
        assert aoi.shapefile_path is None

    def test_repr_contains_point_count(self):
        aoi = AoI(_LAT_DEG, _LON_DEG)
        r = repr(aoi)
        assert 'AoI' in r
        assert '3' in r

    def test_lat_lon_stored_as_float64(self):
        aoi = AoI(list(_LAT_DEG), list(_LON_DEG))
        assert aoi._lat.dtype == np.float64
        assert aoi._lon.dtype == np.float64


# ---------------------------------------------------------------------------
# TestAoIFromRegion
# ---------------------------------------------------------------------------

class TestAoIFromRegion:

    def test_returns_aoi(self):
        aoi = AoI.from_region(lat_min_deg=-10, lat_max_deg=10,
                               lon_min_deg=30,  lon_max_deg=60)
        assert isinstance(aoi, AoI)

    def test_points_in_lat_bounds(self):
        lat_min, lat_max = -10.0, 10.0
        aoi = AoI.from_region(lat_min_deg=lat_min, lat_max_deg=lat_max,
                               lon_min_deg=30,      lon_max_deg=60,
                               point_density=5e12)
        assert np.all(aoi.lat >= lat_min - 1e-6)
        assert np.all(aoi.lat <= lat_max + 1e-6)

    def test_points_in_lon_bounds(self):
        lon_min, lon_max = 30.0, 60.0
        aoi = AoI.from_region(lat_min_deg=-10, lat_max_deg=10,
                               lon_min_deg=lon_min, lon_max_deg=lon_max,
                               point_density=5e12)
        assert np.all(aoi.lon >= lon_min - 1e-6)
        assert np.all(aoi.lon <= lon_max + 1e-6)

    def test_global_region_nonempty(self):
        aoi = AoI.from_region()
        assert len(aoi) > 0

    def test_geometry_none(self):
        aoi = AoI.from_region(lat_min_deg=-10, lat_max_deg=10)
        assert aoi.geometry is None

    def test_shapefile_path_none(self):
        aoi = AoI.from_region(lat_min_deg=-10, lat_max_deg=10)
        assert aoi.shapefile_path is None

    def test_lat_properties_consistent(self):
        aoi = AoI.from_region(lat_min_deg=-10, lat_max_deg=10,
                               lon_min_deg=30,  lon_max_deg=60)
        np.testing.assert_allclose(aoi.lat, np.degrees(aoi.lat_rad))
        np.testing.assert_allclose(aoi.lon, np.degrees(aoi.lon_rad))

    def test_denser_gives_more_points(self):
        aoi_coarse = AoI.from_region(lat_min_deg=-10, lat_max_deg=10,
                                      lon_min_deg=30,  lon_max_deg=60,
                                      point_density=2e12)
        aoi_dense  = AoI.from_region(lat_min_deg=-10, lat_max_deg=10,
                                      lon_min_deg=30,  lon_max_deg=60,
                                      point_density=5e11)
        assert len(aoi_dense) > len(aoi_coarse)


# ---------------------------------------------------------------------------
# TestAoIFromShapefile
# ---------------------------------------------------------------------------

class TestAoIFromShapefile:

    def test_returns_aoi(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e12)
        assert isinstance(aoi, AoI)

    def test_points_nonempty(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e12)
        assert len(aoi) > 0

    def test_geometry_set(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e12)
        assert aoi.geometry is not None

    def test_shapefile_path_set(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e12)
        assert aoi.shapefile_path == str(path)

    def test_lat_lon_degrees_range(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e12)
        assert np.all(aoi.lat >= -90.0)
        assert np.all(aoi.lat <=  90.0)
        assert np.all(aoi.lon >= -180.0)
        assert np.all(aoi.lon <=  180.0)

    def test_lat_rad_range(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e12)
        assert np.all(aoi.lat_rad >= -np.pi / 2)
        assert np.all(aoi.lat_rad <=  np.pi / 2)

    def test_lat_lon_consistent(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e12)
        np.testing.assert_allclose(aoi.lat, np.degrees(aoi.lat_rad))
        np.testing.assert_allclose(aoi.lon, np.degrees(aoi.lon_rad))

    def test_repr_contains_shapefile(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e12)
        assert 'shapefile' in repr(aoi)


# ---------------------------------------------------------------------------
# TestAoIFromGeography
# ---------------------------------------------------------------------------

class TestAoIFromGeography:

    def test_returns_aoi(self):
        aoi = AoI.from_geography('Canada')
        assert isinstance(aoi, AoI)

    def test_nonempty(self):
        aoi = AoI.from_geography('Canada')
        assert len(aoi) > 0

    def test_geometry_set(self):
        aoi = AoI.from_geography('Canada')
        assert aoi.geometry is not None

    def test_shapefile_path_none(self):
        aoi = AoI.from_geography('Canada')
        assert aoi.shapefile_path is None

    def test_lat_lon_degrees(self):
        aoi = AoI.from_geography('Canada')
        assert np.all(aoi.lat >= -90.0) and np.all(aoi.lat <= 90.0)
        assert np.all(aoi.lon >= -180.0) and np.all(aoi.lon <= 180.0)

    def test_iso_a2(self):
        aoi = AoI.from_geography('CA')
        assert len(aoi) > 0

    def test_subdivision(self):
        aoi = AoI.from_geography('Canada/Quebec')
        assert len(aoi) > 0

    def test_unknown_raises(self):
        import pytest
        with pytest.raises(ValueError):
            AoI.from_geography('Narnia')
