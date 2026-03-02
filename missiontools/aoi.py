from __future__ import annotations

import numpy as np
import numpy.typing as npt


class AoI:
    """Area of interest defined by a sampled point cloud.

    All angles are in degrees at the user-facing interface; internally stored
    as radians.  The ``lat_rad`` / ``lon_rad`` properties expose the radians
    representation for direct use with coverage analysis functions.

    Parameters
    ----------
    lat_deg : array-like
        Sample latitudes (deg), shape ``(M,)``.
    lon_deg : array-like
        Sample longitudes (deg), shape ``(M,)``.

    Examples
    --------
    Direct construction from arrays::

        import numpy as np
        from missiontools import AoI

        lat = np.linspace(-10, 10, 50)
        lon = np.linspace(30, 60, 50)
        aoi = AoI(lat, lon)

    From a rectangular lat/lon band::

        aoi = AoI.from_region(lat_min_deg=-10, lat_max_deg=10,
                               lon_min_deg=30,  lon_max_deg=60)

    From an ESRI Shapefile::

        aoi = AoI.from_shapefile('country.shp')
    """

    def __init__(self, lat_deg: npt.ArrayLike, lon_deg: npt.ArrayLike) -> None:
        self._lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
        self._lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
        self._geometry = None
        self._shapefile_path = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lat(self) -> npt.NDArray[np.float64]:
        """Sample latitudes (deg), shape ``(M,)``."""
        return np.degrees(self._lat)

    @property
    def lon(self) -> npt.NDArray[np.float64]:
        """Sample longitudes (deg), shape ``(M,)``."""
        return np.degrees(self._lon)

    @property
    def lat_rad(self) -> npt.NDArray[np.float64]:
        """Sample latitudes (rad), shape ``(M,)`` — for coverage functions."""
        return self._lat

    @property
    def lon_rad(self) -> npt.NDArray[np.float64]:
        """Sample longitudes (rad), shape ``(M,)`` — for coverage functions."""
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
        return len(self._lat)

    def __repr__(self) -> str:
        suffix = (f', shapefile={self._shapefile_path!r}'
                  if self._shapefile_path else '')
        return f'AoI({len(self)} points{suffix})'

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_region(
            cls,
            lat_min_deg: float | None = None,
            lat_max_deg: float | None = None,
            lon_min_deg: float | None = None,
            lon_max_deg: float | None = None,
            *,
            point_density: float = 1e11,
    ) -> 'AoI':
        """Sample an AoI from a rectangular lat/lon region.

        Delegates to :func:`~missiontools.coverage.sample_region`.

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
            Approximate area per sample point (m²).  Defaults to 1×10¹¹ m².

        Returns
        -------
        AoI
        """
        from .coverage import sample_region

        def _r(v: float | None) -> float | None:
            return np.radians(v) if v is not None else None

        lat_rad, lon_rad = sample_region(
            _r(lat_min_deg), _r(lat_max_deg),
            _r(lon_min_deg), _r(lon_max_deg),
            point_density,
        )
        return cls(np.degrees(lat_rad), np.degrees(lon_rad))

    @classmethod
    def from_shapefile(
            cls,
            path: str,
            *,
            feature_index: int | None = None,
            point_density: float = 1e11,
    ) -> 'AoI':
        """Sample an AoI from an ESRI Shapefile polygon.

        Delegates to :func:`~missiontools.coverage.sample_shapefile` for the
        point cloud and also stores the Shapely geometry for downstream use.

        Parameters
        ----------
        path : str
            Path to the ``.shp`` file.
        feature_index : int | None, optional
            Index of the feature to sample.  ``None`` (default) unions all
            features.
        point_density : float, optional
            Approximate area per sample point (m²).  Defaults to 1×10¹¹ m².

        Returns
        -------
        AoI
            With :attr:`geometry` and :attr:`shapefile_path` populated.
        """
        from .coverage import sample_shapefile
        from .coverage.coverage import _load_shapefile

        lat_rad, lon_rad = sample_shapefile(
            path, feature_index=feature_index, point_density=point_density,
        )
        geom, _ = _load_shapefile(path, feature_index)

        obj = cls(np.degrees(lat_rad), np.degrees(lon_rad))
        obj._geometry = geom
        obj._shapefile_path = str(path)
        return obj

    @classmethod
    def from_geography(
            cls,
            geography: str,
            *,
            point_density: float = 1e11,
    ) -> 'AoI':
        """Sample an AoI from a Natural Earth geography by name or code.

        Delegates to :func:`~missiontools.coverage.sample_geography`.

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
            Approximate area per sample point (m²).  Defaults to 1×10¹¹ m².

        Returns
        -------
        AoI
            With :attr:`geometry` populated.
        """
        from .coverage import sample_geography

        lat_rad, lon_rad, geom = sample_geography(
            geography, point_density=point_density,
        )
        obj = cls(np.degrees(lat_rad), np.degrees(lon_rad))
        obj._geometry = geom
        return obj
