from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _geom_crosses_am(geom) -> bool:
    """Return True if *geom*'s bounding box extends outside [-180, 180] longitude."""
    if geom.is_empty:
        return False
    b = geom.bounds  # (minx, miny, maxx, maxy)
    return b[2] > 180.0 or b[0] < -180.0


class AoI:
    """Area of interest defined by a sampled point cloud.

    All angles are in degrees at the user-facing interface; internally stored
    as radians.  The ``lat_rad`` / ``lon_rad`` properties expose the radians
    representation for direct use with coverage analysis functions.

    AoIs created via :meth:`from_region`, :meth:`from_shapefile`,
    :meth:`from_geography`, or a set operation are **geometry-backed** — they
    store a Shapely geometry and generate their sample points **lazily** on
    first access.  This means composing complex AoIs with set operations
    (``|``, ``&``, ``-``, ``^``) incurs no sampling cost until points are
    actually needed.

    Parameters
    ----------
    lat_deg : array-like
        Sample latitudes (deg), shape ``(M,)``.
    lon_deg : array-like
        Sample longitudes (deg), shape ``(M,)``.

    Notes
    -----
    Directly constructed AoIs (``AoI(lat, lon)``) have no associated geometry
    and cannot participate in set operations.

    **Antimeridian caveat**: set operations between a geometry from a Natural
    Earth shapefile that uses unwrapped longitudes (> 180°, e.g. Russia) and
    a :meth:`from_region` box in [-180, 180] will not behave correctly because
    Shapely treats coordinates as Cartesian.  For most geographies this is not
    an issue.

    Examples
    --------
    Direct construction from arrays::

        import numpy as np
        from missiontools import AoI

        lat = np.linspace(-10, 10, 50)
        lon = np.linspace(30, 60, 50)
        aoi = AoI(lat, lon)

    From a rectangular lat/lon band (lazy — no points generated yet)::

        aoi = AoI.from_region(lat_min_deg=-10, lat_max_deg=10,
                               lon_min_deg=30,  lon_max_deg=60)

    From a Natural Earth geography::

        aoi = AoI.from_geography('Australia')

    Compound AoI via set operations::

        conus = AoI.from_geography("US") - AoI.from_geography("US-AK") \\
                                         - AoI.from_geography("US-HI")

        can_arctic = AoI.from_geography("Canada") & AoI.from_region(lat_min_deg=66)
    """

    def __init__(self, lat_deg: npt.ArrayLike, lon_deg: npt.ArrayLike) -> None:
        self._lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
        self._lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
        self._geometry = None
        self._shapefile_path = None
        self._point_density_km2 = None
        self._crosses_am = False

    # ------------------------------------------------------------------
    # Private lazy-construction classmethod
    # ------------------------------------------------------------------

    @classmethod
    def _from_geometry(
        cls,
        geom,
        crosses_am: bool,
        density_km2: float,
        shapefile_path: str | None = None,
    ) -> "AoI":
        """Construct a lazy AoI backed by a Shapely geometry.

        Points are not sampled until first access via a point-returning property.
        """
        obj = object.__new__(cls)
        obj._lat = None  # lazy: computed on demand
        obj._lon = None
        obj._geometry = geom
        obj._shapefile_path = shapefile_path
        obj._point_density_km2 = density_km2
        obj._crosses_am = crosses_am
        return obj

    # ------------------------------------------------------------------
    # Lazy evaluation
    # ------------------------------------------------------------------

    def _ensure_points(self) -> None:
        """Compute sample points from geometry if they have not been generated yet."""
        if self._lat is not None:
            return
        from .coverage.sampling import sample_from_geometry

        if self._geometry.is_empty:
            self._lat = np.empty(0, dtype=np.float64)
            self._lon = np.empty(0, dtype=np.float64)
        else:
            self._lat, self._lon = sample_from_geometry(
                self._geometry, self._crosses_am, self._point_density_km2
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lat(self) -> npt.NDArray[np.float64]:
        """Sample latitudes (deg), shape ``(M,)``."""
        self._ensure_points()
        return np.degrees(self._lat)

    @property
    def lon(self) -> npt.NDArray[np.float64]:
        """Sample longitudes (deg), shape ``(M,)``."""
        self._ensure_points()
        return np.degrees(self._lon)

    @property
    def lat_rad(self) -> npt.NDArray[np.float64]:
        """Sample latitudes (rad), shape ``(M,)`` — for coverage functions."""
        self._ensure_points()
        return self._lat

    @property
    def lon_rad(self) -> npt.NDArray[np.float64]:
        """Sample longitudes (rad), shape ``(M,)`` — for coverage functions."""
        self._ensure_points()
        return self._lon

    @property
    def geometry(self):
        """Shapely geometry describing the AoI, or ``None`` if unavailable."""
        return self._geometry

    @property
    def shapefile_path(self) -> str | None:
        """Path to the source shapefile, or ``None`` if not constructed from one."""
        return self._shapefile_path

    def __len__(self) -> int:
        self._ensure_points()
        return len(self._lat)

    def __repr__(self) -> str:
        pts = f"{len(self._lat)} points" if self._lat is not None else "not yet sampled"
        if self._shapefile_path:
            return f"AoI({pts}, shapefile={self._shapefile_path!r})"
        if self._geometry is not None:
            return f"AoI({pts}, {type(self._geometry).__name__})"
        return f"AoI({pts})"

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def _require_geometry(self, op: str) -> None:
        if self._geometry is None:
            raise TypeError(
                f"Cannot use '{op}' on an AoI without associated geometry. "
                "Use from_region(), from_shapefile(), from_geography(), "
                "or a set operation to create a geometry-backed AoI."
            )

    def __or__(self, other: "AoI") -> "AoI":
        """Union — all area covered by either AoI."""
        self._require_geometry("|")
        other._require_geometry("|")
        geom = self._geometry.union(other._geometry)
        return AoI._from_geometry(geom, _geom_crosses_am(geom), self._point_density_km2)

    def __and__(self, other: "AoI") -> "AoI":
        """Intersection — area common to both AoIs."""
        self._require_geometry("&")
        other._require_geometry("&")
        geom = self._geometry.intersection(other._geometry)
        return AoI._from_geometry(geom, _geom_crosses_am(geom), self._point_density_km2)

    def __sub__(self, other: "AoI") -> "AoI":
        """Difference — area in this AoI that is not in *other*."""
        self._require_geometry("-")
        other._require_geometry("-")
        geom = self._geometry.difference(other._geometry)
        return AoI._from_geometry(geom, _geom_crosses_am(geom), self._point_density_km2)

    def __xor__(self, other: "AoI") -> "AoI":
        """Symmetric difference — area in either AoI but not both."""
        self._require_geometry("^")
        other._require_geometry("^")
        geom = self._geometry.symmetric_difference(other._geometry)
        return AoI._from_geometry(geom, _geom_crosses_am(geom), self._point_density_km2)

    def __sub__(self, other: "AoI") -> "AoI":
        """Difference — area in this AoI that is not in *other*."""
        self._require_geometry("-")
        other._require_geometry("-")
        geom = self._geometry.difference(other._geometry)
        return AoI._from_geometry(geom, _geom_crosses_am(geom), self._point_density_km2)

    def __xor__(self, other: "AoI") -> "AoI":
        """Symmetric difference — area in either AoI but not both."""
        self._require_geometry("^")
        other._require_geometry("^")
        geom = self._geometry.symmetric_difference(other._geometry)
        return AoI._from_geometry(geom, _geom_crosses_am(geom), self._point_density_km2)

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def _from_radians(cls, lat_rad: npt.NDArray, lon_rad: npt.NDArray) -> "AoI":
        """Construct directly from radian arrays, skipping the deg→rad conversion."""
        obj = object.__new__(cls)
        obj._lat = np.asarray(lat_rad, dtype=np.float64)
        obj._lon = np.asarray(lon_rad, dtype=np.float64)
        obj._geometry = None
        obj._shapefile_path = None
        obj._point_density_km2 = None
        obj._crosses_am = False
        return obj

    @classmethod
    def from_region(
        cls,
        lat_min_deg: float | None = None,
        lat_max_deg: float | None = None,
        lon_min_deg: float | None = None,
        lon_max_deg: float | None = None,
        *,
        point_density: float = 1e5,
    ) -> "AoI":
        """Sample an AoI from a rectangular lat/lon region.

        Points are generated lazily on first access.

        Parameters
        ----------
        lat_min_deg : float | None, optional
            Southern boundary (deg).  ``None`` extends to the South Pole.
        lat_max_deg : float | None, optional
            Northern boundary (deg).  ``None`` extends to the North Pole.
        lon_min_deg : float | None, optional
            Western boundary (deg).  Must be paired with ``lon_max_deg``; ``None``
            (together with ``lon_max_deg=None``) includes all longitudes.
        lon_max_deg : float | None, optional
            Eastern boundary (deg).  May be less than ``lon_min_deg`` for
            anti-meridian-crossing regions.
        point_density : float, optional
            Approximate area per sample point (km²).  Defaults to 1×10⁵ km²
            (~100 000 km² per point).

        Returns
        -------
        AoI
            Geometry-backed, lazily sampled.
        """
        from shapely.geometry import box
        from shapely.ops import unary_union

        lat_min = -90.0 if lat_min_deg is None else float(lat_min_deg)
        lat_max = 90.0 if lat_max_deg is None else float(lat_max_deg)

        if lon_min_deg is None and lon_max_deg is None:
            geom = box(-180.0, lat_min, 180.0, lat_max)
            crosses_am = False
        elif lon_min_deg is not None and lon_max_deg is not None:
            lon_min = float(lon_min_deg)
            lon_max = float(lon_max_deg)
            if lon_min <= lon_max:
                geom = box(lon_min, lat_min, lon_max, lat_max)
                crosses_am = False
            else:
                # antimeridian-crossing: two boxes in normal [-180, 180] coordinates
                geom = unary_union(
                    [
                        box(lon_min, lat_min, 180.0, lat_max),
                        box(-180.0, lat_min, lon_max, lat_max),
                    ]
                )
                crosses_am = False
        else:
            raise ValueError(
                "lon_min_deg and lon_max_deg must both be None or both be specified."
            )

        return cls._from_geometry(geom, crosses_am, point_density)

    @classmethod
    def from_shapefile(
        cls,
        path: str,
        *,
        feature_index: int | None = None,
        point_density: float = 1e5,
    ) -> "AoI":
        """Sample an AoI from an ESRI Shapefile polygon.

        Stores the Shapely geometry; points are generated lazily on first access.

        Parameters
        ----------
        path : str
            Path to the ``.shp`` file.
        feature_index : int | None, optional
            Index of the feature to sample.  ``None`` (default) unions all
            features.
        point_density : float, optional
            Approximate area per sample point (km²).  Defaults to 1×10⁵ km².

        Returns
        -------
        AoI
            With :attr:`geometry` and :attr:`shapefile_path` populated.
        """
        from .coverage import load_shapefile_geometry

        geom, crosses_am = load_shapefile_geometry(path, feature_index)
        return cls._from_geometry(
            geom, crosses_am, point_density, shapefile_path=str(path)
        )

    @classmethod
    def from_geography(
        cls,
        geography: str,
        *,
        point_density: float = 1e5,
    ) -> "AoI":
        """Sample an AoI from a Natural Earth geography by name or code.

        Stores the Shapely geometry; points are generated lazily on first access.

        Parameters
        ----------
        geography : str
            One of:

            - Country name: ``'Canada'`` (case-insensitive)
            - ``'Country/Subdivision'``: ``'Canada/Quebec'``
            - ISO 3166-1 alpha-2: ``'CA'``
            - ISO 3166-1 alpha-3: ``'CAN'``
            - ISO 3166-2: ``'CA-QC'``
        point_density : float, optional
            Approximate area per sample point (km²).  Defaults to 1×10⁵ km².

        Returns
        -------
        AoI
            With :attr:`geometry` populated.
        """
        from .coverage import geography_geometry

        geom, crosses_am = geography_geometry(geography)
        return cls._from_geometry(geom, crosses_am, point_density)
