import numpy as np
import pytest

from missiontools import AoI


# ---------------------------------------------------------------------------
# Helpers / shared data
# ---------------------------------------------------------------------------

_LAT_DEG = np.array([-10.0, 0.0, 10.0])
_LON_DEG = np.array([30.0, 45.0, 60.0])


# Shapefile helper (mirrors test_coverage.py)
def _make_shapefile(tmp_path, rings_lon_lat_deg, name="test"):
    """Write a single-feature Polygon shapefile; return path to .shp."""
    import shapefile as _pyshp  # noqa: PLC0415

    path = str(tmp_path / f"{name}.shp")
    w = _pyshp.Writer(path, shapeType=5)
    w.field("name", "C")

    def _area2(ring):
        n = len(ring)
        return sum(
            ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1]
            for i in range(n)
        )

    # ESRI convention: exterior rings are CW (negative signed area),
    # hole rings are CCW (positive signed area).
    fixed = []
    for i, ring in enumerate(rings_lon_lat_deg):
        pts = list(ring)
        a2 = _area2(pts)
        if i == 0:  # exterior: force CW
            if a2 > 0:
                pts = pts[::-1]
        else:  # hole: force CCW
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
        assert "AoI" in r
        assert "3" in r

    def test_lat_lon_stored_as_float64(self):
        aoi = AoI(list(_LAT_DEG), list(_LON_DEG))
        assert aoi._lat.dtype == np.float64
        assert aoi._lon.dtype == np.float64


# ---------------------------------------------------------------------------
# TestAoIFromRegion
# ---------------------------------------------------------------------------


class TestAoIFromRegion:
    def test_returns_aoi(self):
        aoi = AoI.from_region(
            lat_min_deg=-10, lat_max_deg=10, lon_min_deg=30, lon_max_deg=60
        )
        assert isinstance(aoi, AoI)

    def test_lazy_before_access(self):
        aoi = AoI.from_region(
            lat_min_deg=-10, lat_max_deg=10, lon_min_deg=30, lon_max_deg=60
        )
        assert aoi._lat is None

    def test_lazy_after_access(self):
        aoi = AoI.from_region(
            lat_min_deg=-10, lat_max_deg=10, lon_min_deg=30, lon_max_deg=60
        )
        _ = aoi.lat_rad
        assert aoi._lat is not None

    def test_geometry_set(self):
        aoi = AoI.from_region(lat_min_deg=-10, lat_max_deg=10)
        assert aoi.geometry is not None

    def test_geometry_is_box(self):
        from shapely.geometry import Polygon

        aoi = AoI.from_region(
            lat_min_deg=-10, lat_max_deg=10, lon_min_deg=30, lon_max_deg=60
        )
        assert isinstance(aoi.geometry, Polygon)

    def test_points_in_lat_bounds(self):
        lat_min, lat_max = -10.0, 10.0
        aoi = AoI.from_region(
            lat_min_deg=lat_min,
            lat_max_deg=lat_max,
            lon_min_deg=30,
            lon_max_deg=60,
            point_density=5e6,
        )
        assert np.all(aoi.lat >= lat_min - 1e-6)
        assert np.all(aoi.lat <= lat_max + 1e-6)

    def test_points_in_lon_bounds(self):
        lon_min, lon_max = 30.0, 60.0
        aoi = AoI.from_region(
            lat_min_deg=-10,
            lat_max_deg=10,
            lon_min_deg=lon_min,
            lon_max_deg=lon_max,
            point_density=5e6,
        )
        assert np.all(aoi.lon >= lon_min - 1e-6)
        assert np.all(aoi.lon <= lon_max + 1e-6)

    def test_global_region_nonempty(self):
        aoi = AoI.from_region()
        assert len(aoi) > 0

    def test_shapefile_path_none(self):
        aoi = AoI.from_region(lat_min_deg=-10, lat_max_deg=10)
        assert aoi.shapefile_path is None

    def test_lat_properties_consistent(self):
        aoi = AoI.from_region(
            lat_min_deg=-10, lat_max_deg=10, lon_min_deg=30, lon_max_deg=60
        )
        np.testing.assert_allclose(aoi.lat, np.degrees(aoi.lat_rad))
        np.testing.assert_allclose(aoi.lon, np.degrees(aoi.lon_rad))

    def test_denser_gives_more_points(self):
        # The Fibonacci-sphere floor is 5 000 global points, reached when
        # density > ~1.73e5 km²/pt.  One value must be finer than that floor
        # to see a count difference (see MEMORY.md notes on Fibonacci floor).
        aoi_coarse = AoI.from_region(
            lat_min_deg=-10,
            lat_max_deg=10,
            lon_min_deg=30,
            lon_max_deg=60,
            point_density=2e5,
        )  # hits floor → ~73 pts
        aoi_dense = AoI.from_region(
            lat_min_deg=-10,
            lat_max_deg=10,
            lon_min_deg=30,
            lon_max_deg=60,
            point_density=5e4,
        )  # above floor → more pts
        assert len(aoi_dense) > len(aoi_coarse)


# ---------------------------------------------------------------------------
# TestAoIFromShapefile
# ---------------------------------------------------------------------------


class TestAoIFromShapefile:
    def test_returns_aoi(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e6)
        assert isinstance(aoi, AoI)

    def test_lazy_before_access(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e6)
        assert aoi._lat is None

    def test_points_nonempty(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e6)
        assert len(aoi) > 0

    def test_geometry_set(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e6)
        assert aoi.geometry is not None

    def test_shapefile_path_set(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e6)
        assert aoi.shapefile_path == str(path)

    def test_lat_lon_degrees_range(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e6)
        assert np.all(aoi.lat >= -90.0)
        assert np.all(aoi.lat <= 90.0)
        assert np.all(aoi.lon >= -180.0)
        assert np.all(aoi.lon <= 180.0)

    def test_lat_rad_range(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e6)
        assert np.all(aoi.lat_rad >= -np.pi / 2)
        assert np.all(aoi.lat_rad <= np.pi / 2)

    def test_lat_lon_consistent(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e6)
        np.testing.assert_allclose(aoi.lat, np.degrees(aoi.lat_rad))
        np.testing.assert_allclose(aoi.lon, np.degrees(aoi.lon_rad))

    def test_repr_contains_shapefile(self, tmp_path):
        path = _make_shapefile(tmp_path, _BOX_10E)
        aoi = AoI.from_shapefile(path, point_density=5e6)
        assert "shapefile" in repr(aoi)


# ---------------------------------------------------------------------------
# TestAoIFromGeography
# ---------------------------------------------------------------------------


class TestAoIFromGeography:
    def test_returns_aoi(self):
        aoi = AoI.from_geography("Canada")
        assert isinstance(aoi, AoI)

    def test_lazy_before_access(self):
        aoi = AoI.from_geography("Canada")
        assert aoi._lat is None

    def test_nonempty(self):
        aoi = AoI.from_geography("Canada")
        assert len(aoi) > 0

    def test_geometry_set(self):
        aoi = AoI.from_geography("Canada")
        assert aoi.geometry is not None

    def test_shapefile_path_none(self):
        aoi = AoI.from_geography("Canada")
        assert aoi.shapefile_path is None

    def test_lat_lon_degrees(self):
        aoi = AoI.from_geography("Canada")
        assert np.all(aoi.lat >= -90.0) and np.all(aoi.lat <= 90.0)
        assert np.all(aoi.lon >= -180.0) and np.all(aoi.lon <= 180.0)

    def test_iso_a2(self):
        aoi = AoI.from_geography("CA")
        assert len(aoi) > 0

    def test_subdivision(self):
        aoi = AoI.from_geography("Canada/Quebec")
        assert len(aoi) > 0

    def test_unknown_raises(self):
        import pytest

        with pytest.raises(ValueError):
            AoI.from_geography("Narnia")


# ---------------------------------------------------------------------------
# TestAoISetOps
# ---------------------------------------------------------------------------


class TestAoISetOps:
    # --- Helpers ---

    @pytest.fixture
    def europe(self):
        return AoI.from_region(
            lat_min_deg=35, lat_max_deg=72, lon_min_deg=-10, lon_max_deg=40
        )

    @pytest.fixture
    def north_europe(self):
        return AoI.from_region(
            lat_min_deg=55, lat_max_deg=72, lon_min_deg=-10, lon_max_deg=40
        )

    @pytest.fixture
    def australia(self):
        return AoI.from_region(
            lat_min_deg=-40, lat_max_deg=-10, lon_min_deg=113, lon_max_deg=154
        )

    # --- Union ---

    def test_union_returns_aoi(self, europe, australia):
        result = europe | australia
        assert isinstance(result, AoI)

    def test_union_is_lazy(self, europe, australia):
        result = europe | australia
        assert result._lat is None

    def test_union_geometry(self, europe, australia):
        result = europe | australia
        assert result.geometry is not None
        assert not result.geometry.is_empty

    def test_union_has_more_points(self, europe, australia):
        result = europe | australia
        assert len(result) >= len(europe)

    # --- Intersection ---

    def test_intersection_returns_aoi(self, europe, north_europe):
        result = europe & north_europe
        assert isinstance(result, AoI)

    def test_intersection_is_lazy(self, europe, north_europe):
        result = europe & north_europe
        assert result._lat is None

    def test_intersection_geometry(self, europe, north_europe):
        result = europe & north_europe
        assert not result.geometry.is_empty

    def test_intersection_smaller_than_input(self, europe, north_europe):
        result = europe & north_europe
        assert len(result) < len(europe)

    # --- Difference ---

    def test_difference_returns_aoi(self, europe, north_europe):
        result = europe - north_europe
        assert isinstance(result, AoI)

    def test_difference_is_lazy(self, europe, north_europe):
        result = europe - north_europe
        assert result._lat is None

    def test_difference_geometry(self, europe, north_europe):
        result = europe - north_europe
        assert not result.geometry.is_empty

    def test_difference_smaller_than_input(self, europe, north_europe):
        result = europe - north_europe
        assert len(result) < len(europe)
        assert len(result) > 0

    # --- Symmetric difference ---

    def test_symmetric_difference_returns_aoi(self, europe, north_europe):
        result = europe ^ north_europe
        assert isinstance(result, AoI)

    def test_symmetric_difference_geometry(self, europe, north_europe):
        result = europe ^ north_europe
        assert not result.geometry.is_empty

    # --- Chained operations ---

    def test_chained_difference(self):
        """Mirrors the CONUS example from the docstring."""
        us = AoI.from_geography("US")
        ak = AoI.from_geography("US-AK")
        hi = AoI.from_geography("US-HI")
        conus = us - ak - hi
        assert isinstance(conus, AoI)
        assert conus._lat is None  # still lazy
        assert len(conus) > 0
        assert len(conus) < len(us)

    def test_intersection_with_region(self):
        """Mirrors the Canadian Arctic example from the docstring."""
        can = AoI.from_geography("Canada")
        arctic_band = AoI.from_region(lat_min_deg=66)
        can_arctic = can & arctic_band
        assert isinstance(can_arctic, AoI)
        assert len(can_arctic) > 0
        assert len(can_arctic) < len(can)

    # --- Empty intersection ---

    def test_empty_intersection_gives_zero_len(self, europe, australia):
        empty = europe & australia
        assert isinstance(empty, AoI)
        assert len(empty) == 0
        assert empty.geometry.is_empty

    def test_empty_intersection_arrays_are_empty(self, europe, australia):
        empty = europe & australia
        assert len(empty.lat_rad) == 0
        assert len(empty.lon_rad) == 0

    # --- Density inheritance ---

    def test_density_from_first_operand(self):
        a = AoI.from_region(
            lat_min_deg=0,
            lat_max_deg=30,
            lon_min_deg=0,
            lon_max_deg=30,
            point_density=1e4,
        )
        b = AoI.from_region(
            lat_min_deg=10,
            lat_max_deg=40,
            lon_min_deg=10,
            lon_max_deg=40,
            point_density=9e4,
        )
        result = a | b
        assert result._point_density_km2 == a._point_density_km2

    # --- TypeError for geometry-less AoI ---

    def test_set_op_requires_geometry_lhs(self):
        eager = AoI(_LAT_DEG, _LON_DEG)
        geom = AoI.from_region(lat_min_deg=-10, lat_max_deg=10)
        with pytest.raises(TypeError, match="geometry"):
            _ = eager | geom

    def test_set_op_requires_geometry_rhs(self):
        eager = AoI(_LAT_DEG, _LON_DEG)
        geom = AoI.from_region(lat_min_deg=-10, lat_max_deg=10)
        with pytest.raises(TypeError, match="geometry"):
            _ = geom | eager
