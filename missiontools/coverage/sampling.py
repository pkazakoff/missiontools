import re as _re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from matplotlib.path import Path as _MplPath

from ..orbit.constants import EARTH_MEAN_RADIUS

_GEODATA_DIR = Path(__file__).parent / "geodata"
_NE_ADM0 = _GEODATA_DIR / "ne_map_units" / "ne_50m_admin_0_map_units.shp"
_NE_ADM1 = _GEODATA_DIR / "ne_states_provinces" / "ne_50m_admin_1_states_provinces.shp"


def _fibonacci_sphere(n: int) -> tuple[npt.NDArray, npt.NDArray]:
    if n == 1:
        return np.array([0.0]), np.array([0.0])
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n, dtype=np.float64)
    lat = np.arcsin(np.clip(2.0 * i / (n - 1) - 1.0, -1.0, 1.0))
    lon = (2.0 * np.pi * i / phi) % (2.0 * np.pi) - np.pi
    return lat, lon


def _pip(
    polygon: npt.NDArray, lat: npt.NDArray, lon: npt.NDArray
) -> npt.NDArray[np.bool_]:
    path = _MplPath(polygon[:, ::-1])
    return path.contains_points(np.column_stack([lon, lat]))


def sample_aoi(
    polygon: npt.NDArray[np.floating],
    n: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon must have shape (V, 2) with columns [lat, lon]")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    n_pilot = max(n * 10, 5_000)
    lat_p, lon_p = _fibonacci_sphere(n_pilot)
    frac = float(_pip(polygon, lat_p, lon_p).mean())

    if frac < 1e-6:
        raise ValueError(
            "AoI polygon encloses too few global lattice points — "
            "check that coordinates are in radians and the polygon is not "
            "degenerate."
        )

    n_global = int(np.ceil(n / frac * 1.3))
    lat_all, lon_all = _fibonacci_sphere(n_global)
    inside = _pip(polygon, lat_all, lon_all)
    lat_in = lat_all[inside]
    lon_in = lon_all[inside]

    if len(lat_in) > n:
        idx = np.round(np.linspace(0, len(lat_in) - 1, n)).astype(int)
        lat_in = lat_in[idx]
        lon_in = lon_in[idx]

    return lat_in, lon_in


def sample_region(
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    point_density: float = 1e5,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    if (lon_min is None) != (lon_max is None):
        raise ValueError(
            "lon_min and lon_max must both be None (all longitudes) or both "
            "be specified; got lon_min={} lon_max={}".format(lon_min, lon_max)
        )

    if point_density <= 0:
        raise ValueError(f"point_density must be positive, got {point_density}")

    lat_lo = float(lat_min) if lat_min is not None else -np.pi / 2.0
    lat_hi = float(lat_max) if lat_max is not None else np.pi / 2.0
    full_lon = lon_min is None

    if lat_lo >= lat_hi:
        raise ValueError(
            f"lat_min ({lat_lo:.6f} rad) must be less than lat_max ({lat_hi:.6f} rad)"
        )

    lon_lo = float(lon_min) if lon_min is not None else 0.0
    lon_hi = float(lon_max) if lon_max is not None else 0.0
    antimeridian = (not full_lon) and (lon_lo > lon_hi)

    if full_lon:
        lon_frac = 1.0
    elif antimeridian:
        lon_frac = (2.0 * np.pi - (lon_lo - lon_hi)) / (2.0 * np.pi)
    else:
        lon_frac = (lon_hi - lon_lo) / (2.0 * np.pi)

    area = (
        4.0
        * np.pi
        * EARTH_MEAN_RADIUS**2
        * (np.sin(lat_hi) - np.sin(lat_lo))
        / 2.0
        * lon_frac
    )

    pd_m2 = point_density * 1e6
    n = max(1, int(np.round(area / pd_m2)))

    area_fraction = area / (4.0 * np.pi * EARTH_MEAN_RADIUS**2)
    n_global = max(n * 5, int(np.ceil(n / area_fraction * 1.3)))
    lat_all, lon_all = _fibonacci_sphere(n_global)

    lat_ok = (lat_all >= lat_lo) & (lat_all <= lat_hi)

    if full_lon:
        lon_ok = np.ones(n_global, dtype=np.bool_)
    elif antimeridian:
        lon_ok = (lon_all >= lon_lo) | (lon_all <= lon_hi)
    else:
        lon_ok = (lon_all >= lon_lo) & (lon_all <= lon_hi)

    lat_in = lat_all[lat_ok & lon_ok]
    lon_in = lon_all[lat_ok & lon_ok]

    if len(lat_in) > n:
        idx = np.round(np.linspace(0, len(lat_in) - 1, n)).astype(int)
        lat_in = lat_in[idx]
        lon_in = lon_in[idx]

    return lat_in, lon_in


_SHP_POLYGON_TYPES = frozenset({5, 15, 25})


def _unwrap_ring(coords: list) -> tuple[list, bool]:
    lons_raw = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    lons_u = [lons_raw[0]]
    crosses = False
    for j in range(1, len(lons_raw)):
        dl = lons_raw[j] - lons_u[-1]
        if dl > 180.0:
            lons_u.append(lons_raw[j] - 360.0)
            crosses = True
        elif dl < -180.0:
            lons_u.append(lons_raw[j] + 360.0)
            crosses = True
        else:
            lons_u.append(lons_raw[j])
    return list(zip(lons_u, lats)), crosses


def load_shapefile_geometry(path, feature_index):
    import shapefile as _pyshp
    from shapely.geometry import shape as _shape, Polygon as _Polygon
    from shapely.geometry import MultiPolygon as _MultiPolygon
    from shapely.ops import unary_union as _unary_union

    sf = _pyshp.Reader(str(path))

    if feature_index is not None:
        raw_shapes = [sf.shape(feature_index)]
    else:
        raw_shapes = sf.shapes()

    crosses_am = False
    geoms = []

    for shp in raw_shapes:
        if shp.shapeType not in _SHP_POLYGON_TYPES:
            continue

        geo = _shape(shp.__geo_interface__)
        polys = list(geo.geoms) if isinstance(geo, _MultiPolygon) else [geo]

        for poly in polys:
            ext_coords, cam = _unwrap_ring(list(poly.exterior.coords))
            crosses_am = crosses_am or cam

            int_coords_list = []
            for interior in poly.interiors:
                ic, cam = _unwrap_ring(list(interior.coords))
                crosses_am = crosses_am or cam
                int_coords_list.append(ic)

            geoms.append(_Polygon(ext_coords, int_coords_list))

    if not geoms:
        raise ValueError(
            "No polygon features found in the shapefile.  "
            "Only shapeTypes 5 (Polygon), 15 (PolygonZ), and 25 (PolygonM) "
            "are supported."
        )

    geom = _unary_union(geoms) if len(geoms) > 1 else geoms[0]
    return geom, crosses_am


def sample_from_geometry(
    geom,
    crosses_am: bool,
    point_density: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    import shapely as _shapely

    _SPHERE_AREA = 4.0 * np.pi * EARTH_MEAN_RADIUS**2
    pd_m2 = point_density * 1e6
    n_global = max(5_000, int(np.ceil(_SPHERE_AREA / pd_m2 * 1.3)))
    lat_r, lon_r = _fibonacci_sphere(n_global)

    lon_deg = np.degrees(lon_r)
    lat_deg = np.degrees(lat_r)

    inside = _shapely.contains_xy(geom, lon_deg, lat_deg)
    if crosses_am:
        inside |= _shapely.contains_xy(geom, lon_deg + 360.0, lat_deg)
        inside |= _shapely.contains_xy(geom, lon_deg - 360.0, lat_deg)

    lat_in = lat_r[inside]
    lon_in = lon_r[inside]

    if len(lat_in) == 0:
        raise ValueError(
            "No sample points fell inside the geometry — "
            "check that coordinates are in geographic degrees (WGS84 / EPSG:4326)."
        )

    return lat_in, lon_in


def sample_shapefile(
    path: str,
    *,
    feature_index: int | None = None,
    point_density: float = 1e5,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    if point_density <= 0:
        raise ValueError(f"point_density must be positive, got {point_density}")

    geom, crosses_am = load_shapefile_geometry(path, feature_index)
    return sample_from_geometry(geom, crosses_am, point_density)


def _load_ne_features(
    path: str,
    indices: list[int],
) -> tuple:
    from shapely.ops import unary_union as _unary_union

    geoms: list = []
    crosses_am = False
    for i in indices:
        g, cam = load_shapefile_geometry(path, i)
        geoms.append(g)
        crosses_am = crosses_am or cam

    if not geoms:
        raise ValueError("No features matched the requested indices.")

    return (_unary_union(geoms) if len(geoms) > 1 else geoms[0]), crosses_am


def _find_ne_indices(geography: str) -> tuple[str, list[int]]:
    import shapefile as _pyshp

    g = geography.strip()

    if "/" in g:
        country, subdivision = (s.strip() for s in g.split("/", 1))
        cl = country.lower()
        sl = subdivision.lower()
        sf1 = _pyshp.Reader(str(_NE_ADM1))
        idx = [
            i
            for i, r in enumerate(sf1.records())
            if r.as_dict()["admin"].lower() == cl
            and (
                r.as_dict()["name"].lower() == sl
                or r.as_dict()["name_en"].lower() == sl
            )
        ]
        if not idx:
            raise ValueError(
                f"Subdivision {subdivision!r} not found in {country!r}. "
                f"Sub-national (admin-1) data is available only for: "
                f"Australia, Brazil, Canada, China, India, Indonesia, "
                f"Russia, South Africa, United States of America."
            )
        return str(_NE_ADM1), idx

    if _re.match(r"^[A-Z]{2}-[A-Z0-9]{1,3}$", g):
        sf1 = _pyshp.Reader(str(_NE_ADM1))
        idx = [i for i, r in enumerate(sf1.records()) if r.as_dict()["iso_3166_2"] == g]
        if idx:
            return str(_NE_ADM1), idx

    if len(g) == 2 and g.isupper():
        sf0 = _pyshp.Reader(str(_NE_ADM0))
        idx = [i for i, r in enumerate(sf0.records()) if r.as_dict()["ISO_A2"] == g]
        if idx:
            return str(_NE_ADM0), idx

    if len(g) == 3 and g.isupper():
        sf0 = _pyshp.Reader(str(_NE_ADM0))
        idx = [i for i, r in enumerate(sf0.records()) if r.as_dict()["ISO_A3"] == g]
        if idx:
            return str(_NE_ADM0), idx

    gl = g.lower()

    sf0 = _pyshp.Reader(str(_NE_ADM0))
    idx = [i for i, r in enumerate(sf0.records()) if r.as_dict()["NAME"].lower() == gl]
    if idx:
        return str(_NE_ADM0), idx

    sf1 = _pyshp.Reader(str(_NE_ADM1))
    idx = [
        i
        for i, r in enumerate(sf1.records())
        if r.as_dict()["name"].lower() == gl or r.as_dict()["name_en"].lower() == gl
    ]
    if idx:
        return str(_NE_ADM1), idx

    raise ValueError(
        f"Geography not found: {geography!r}. "
        f"Accepted formats: country name ('Canada'), "
        f"'Country/Subdivision' ('Canada/Quebec'), "
        f"ISO A2 ('CA'), ISO A3 ('CAN'), ISO 3166-2 ('CA-QC')."
    )


def sample_geography(
    geography: str,
    *,
    point_density: float = 1e5,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], object]:
    if point_density <= 0:
        raise ValueError(f"point_density must be positive, got {point_density}")

    path, indices = _find_ne_indices(geography)
    geom, crosses_am = _load_ne_features(path, indices)
    lat, lon = sample_from_geometry(geom, crosses_am, point_density)
    return lat, lon, geom


def geography_geometry(geography: str) -> tuple[object, bool]:
    path, indices = _find_ne_indices(geography)
    return _load_ne_features(path, indices)
