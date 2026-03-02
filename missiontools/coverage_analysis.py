"""
missiontools.coverage_analysis
================================
Coverage analysis object bundling an AoI, sensors, and observation constraints.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .aoi import AoI
from .sensor import Sensor
from .coverage import (
    coverage_fraction  as _coverage_fraction,
    revisit_time       as _revisit_time,
    pointwise_coverage as _pointwise_coverage,
    access_pointwise   as _access_pointwise,
    revisit_pointwise  as _revisit_pointwise,
)

_DEFAULT_STEP = np.timedelta64(30, 's')


class Coverage:
    """Coverage analysis for an area of interest observed by a sensor.

    The sensor must be attached to a :class:`~missiontools.Spacecraft` via
    :meth:`~missiontools.Spacecraft.add_sensor` before constructing a
    ``Coverage`` object.  The spacecraft orbit and propagator are derived
    automatically from the sensor's back-reference.

    Parameters
    ----------
    aoi : AoI
        Area of interest containing ground sample points.
    sensors : list[Sensor]
        List containing exactly **one** :class:`~missiontools.Sensor`.
        Multiple sensors are not yet supported.
    el_min_deg : float, optional
        Minimum ground elevation angle (degrees) for a ground point to be
        considered in view.  Default 0 (horizon).
    sza_max_deg : float | None, optional
        Maximum solar zenith angle (degrees) for access (daytime constraint).
        ``None`` disables the constraint.
    sza_min_deg : float | None, optional
        Minimum solar zenith angle (degrees) for access (nighttime constraint).
        ``None`` disables the constraint.

    Notes
    -----
    All five coverage methods evaluate the sensor's LVLH boresight at
    *t_start* and assume it is constant in the LVLH frame for the full
    analysis window.  This is exact for sensors with fixed-LVLH pointing
    (nadir, fixed tilt, body-mounted on a nadir spacecraft) and an
    approximation for time-varying pointing modes.

    Constellation coverage (multiple sensors on different spacecraft) is not
    yet supported.

    Examples
    --------
    ::

        import numpy as np
        from missiontools import Spacecraft, Sensor, AoI, Coverage

        sc     = Spacecraft(a=6_771_000., e=0., i=np.radians(51.6),
                            raan=0., arg_p=0., ma=0.,
                            epoch=np.datetime64('2025-01-01', 'us'))
        sensor = Sensor(30.0, body_vector=[0, 0, 1])
        sc.add_sensor(sensor)

        aoi = AoI.from_region(-60, 60, -180, 180)
        cov = Coverage(aoi, [sensor], el_min_deg=5.0)

        result = cov.coverage_fraction(
            np.datetime64('2025-01-01', 'us'),
            np.datetime64('2025-01-02', 'us'),
        )
        print(result['final_cumulative'])
    """

    def __init__(
            self,
            aoi: AoI,
            sensors: list,
            *,
            el_min_deg: float = 0.0,
            sza_max_deg: float | None = None,
            sza_min_deg: float | None = None,
    ):
        # --- validate aoi ---------------------------------------------------
        if not isinstance(aoi, AoI):
            raise TypeError(
                f"aoi must be an AoI instance, got {type(aoi).__name__!r}"
            )

        # --- validate sensors -----------------------------------------------
        if not isinstance(sensors, list) or len(sensors) == 0:
            raise ValueError("sensors must be a non-empty list of Sensor objects")
        for s in sensors:
            if not isinstance(s, Sensor):
                raise TypeError(
                    f"Each element of sensors must be a Sensor instance, "
                    f"got {type(s).__name__!r}"
                )
        if len(sensors) > 1:
            raise NotImplementedError(
                "Multiple sensors are not yet supported. "
                "Pass a list with exactly one Sensor."
            )

        # --- validate sensor is attached to a spacecraft --------------------
        sensor = sensors[0]
        if sensor.spacecraft is None:
            raise RuntimeError(
                "Sensor must be attached to a Spacecraft via add_sensor() "
                "before use with Coverage."
            )

        # --- validate constraints -------------------------------------------
        if float(el_min_deg) < 0.0:
            raise ValueError(
                f"el_min_deg must be >= 0, got {el_min_deg}"
            )

        # --- store state ----------------------------------------------------
        self._aoi         = aoi
        self._sensors     = list(sensors)
        self._el_min_deg  = float(el_min_deg)
        self._sza_max_deg = float(sza_max_deg) if sza_max_deg is not None else None
        self._sza_min_deg = float(sza_min_deg) if sza_min_deg is not None else None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def aoi(self) -> AoI:
        """Area of interest."""
        return self._aoi

    @property
    def sensors(self) -> list:
        """Attached sensors (read-only copy)."""
        return list(self._sensors)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fov_params(
            self,
            t_start: np.datetime64,
    ) -> tuple[npt.NDArray[np.floating], float]:
        """Return ``(fov_pointing_lvlh, fov_half_angle_rad)`` for the single sensor.

        The boresight is evaluated at *t_start* and assumed constant in the
        LVLH frame for the duration of the analysis window.
        """
        sensor = self._sensors[0]
        sc     = sensor.spacecraft
        state  = sc.propagate(
            t_start,
            t_start + np.timedelta64(1, 's'),
            np.timedelta64(1, 's'),
        )
        r, v, t = state['r'][0], state['v'][0], state['t'][0]
        return sensor.pointing_lvlh(r, v, t), sensor.half_angle_rad

    def _call_kwargs(
            self,
            t_start: np.datetime64,
            alt: float,
            max_step: np.timedelta64,
            batch_size: int,
    ) -> dict:
        """Shared keyword arguments for the functional coverage API."""
        fov_pointing, fov_half_angle = self._fov_params(t_start)
        sc = self._sensors[0].spacecraft
        return dict(
            propagator_type   = sc.propagator_type,
            el_min            = np.radians(self._el_min_deg),
            sza_max           = (np.radians(self._sza_max_deg)
                                 if self._sza_max_deg is not None else None),
            sza_min           = (np.radians(self._sza_min_deg)
                                 if self._sza_min_deg is not None else None),
            fov_pointing_lvlh = fov_pointing,
            fov_half_angle    = fov_half_angle,
            alt               = alt,
            max_step          = max_step,
            batch_size        = batch_size,
        )

    # ------------------------------------------------------------------
    # Coverage methods
    # ------------------------------------------------------------------

    def coverage_fraction(
            self,
            t_start: np.datetime64,
            t_end:   np.datetime64,
            *,
            alt:        float          = 0.0,
            max_step:   np.timedelta64 = _DEFAULT_STEP,
            batch_size: int            = 1_000,
    ) -> dict:
        """Instantaneous and cumulative coverage fraction over time.

        Parameters
        ----------
        t_start : np.datetime64
            Start of the analysis window.
        t_end : np.datetime64
            End of the analysis window (inclusive).
        alt : float, optional
            Altitude of ground targets above WGS84 (m).  Default 0.
        max_step : np.timedelta64, optional
            Maximum propagation step (default 30 s).
        batch_size : int, optional
            Propagation batch size (default 1000).

        Returns
        -------
        dict
            ``'t'``, ``'fraction'``, ``'cumulative'``, ``'mean_fraction'``,
            ``'final_cumulative'``.  See
            :func:`~missiontools.coverage.coverage_fraction`.
        """
        sc = self._sensors[0].spacecraft
        return _coverage_fraction(
            self._aoi.lat_rad, self._aoi.lon_rad,
            sc.keplerian_params,
            t_start, t_end,
            **self._call_kwargs(t_start, alt, max_step, batch_size),
        )

    def revisit_time(
            self,
            t_start: np.datetime64,
            t_end:   np.datetime64,
            *,
            alt:        float          = 0.0,
            max_step:   np.timedelta64 = _DEFAULT_STEP,
            batch_size: int            = 1_000,
    ) -> dict:
        """Per-point revisit statistics.

        Parameters
        ----------
        t_start : np.datetime64
            Start of the analysis window.
        t_end : np.datetime64
            End of the analysis window (inclusive).
        alt : float, optional
            Altitude of ground targets above WGS84 (m).  Default 0.
        max_step : np.timedelta64, optional
            Maximum propagation step (default 30 s).
        batch_size : int, optional
            Propagation batch size (default 1000).

        Returns
        -------
        dict
            ``'max_revisit'``, ``'mean_revisit'``, ``'global_max'``,
            ``'global_mean'``.  See
            :func:`~missiontools.coverage.revisit_time`.
        """
        sc = self._sensors[0].spacecraft
        return _revisit_time(
            self._aoi.lat_rad, self._aoi.lon_rad,
            sc.keplerian_params,
            t_start, t_end,
            **self._call_kwargs(t_start, alt, max_step, batch_size),
        )

    def pointwise_coverage(
            self,
            t_start: np.datetime64,
            t_end:   np.datetime64,
            *,
            alt:        float          = 0.0,
            max_step:   np.timedelta64 = _DEFAULT_STEP,
            batch_size: int            = 1_000,
    ) -> dict:
        """Raw per-timestep visibility matrix.

        Parameters
        ----------
        t_start : np.datetime64
            Start of the analysis window.
        t_end : np.datetime64
            End of the analysis window (inclusive).
        alt : float, optional
            Altitude of ground targets above WGS84 (m).  Default 0.
        max_step : np.timedelta64, optional
            Maximum propagation step (default 30 s).
        batch_size : int, optional
            Propagation batch size (default 1000).

        Returns
        -------
        dict
            ``'t'``, ``'lat'``, ``'lon'``, ``'alt'``, ``'visible'``.  See
            :func:`~missiontools.coverage.pointwise_coverage`.
        """
        sc = self._sensors[0].spacecraft
        return _pointwise_coverage(
            self._aoi.lat_rad, self._aoi.lon_rad,
            sc.keplerian_params,
            t_start, t_end,
            **self._call_kwargs(t_start, alt, max_step, batch_size),
        )

    def access_pointwise(
            self,
            t_start: np.datetime64,
            t_end:   np.datetime64,
            *,
            alt:        float          = 0.0,
            max_step:   np.timedelta64 = _DEFAULT_STEP,
            batch_size: int            = 1_000,
    ) -> list:
        """Per-point access intervals (AOS / LOS pairs).

        Parameters
        ----------
        t_start : np.datetime64
            Start of the analysis window.
        t_end : np.datetime64
            End of the analysis window (inclusive).
        alt : float, optional
            Altitude of ground targets above WGS84 (m).  Default 0.
        max_step : np.timedelta64, optional
            Maximum propagation step (default 30 s).
        batch_size : int, optional
            Propagation batch size (default 1000).

        Returns
        -------
        list[list[tuple[np.datetime64, np.datetime64]]]
            ``result[m]`` is a list of ``(AOS, LOS)`` pairs for ground point *m*.
            See :func:`~missiontools.coverage.access_pointwise`.
        """
        sc = self._sensors[0].spacecraft
        return _access_pointwise(
            self._aoi.lat_rad, self._aoi.lon_rad,
            sc.keplerian_params,
            t_start, t_end,
            **self._call_kwargs(t_start, alt, max_step, batch_size),
        )

    def revisit_pointwise(
            self,
            t_start: np.datetime64,
            t_end:   np.datetime64,
            *,
            alt:        float          = 0.0,
            max_step:   np.timedelta64 = _DEFAULT_STEP,
            batch_size: int            = 1_000,
    ) -> list:
        """Per-point revisit gap arrays.

        Parameters
        ----------
        t_start : np.datetime64
            Start of the analysis window.
        t_end : np.datetime64
            End of the analysis window (inclusive).
        alt : float, optional
            Altitude of ground targets above WGS84 (m).  Default 0.
        max_step : np.timedelta64, optional
            Maximum propagation step (default 30 s).
        batch_size : int, optional
            Propagation batch size (default 1000).

        Returns
        -------
        list[np.ndarray of timedelta64]
            ``result[m]`` is an array of LOS-to-AOS gap durations for ground
            point *m*.  See :func:`~missiontools.coverage.revisit_pointwise`.
        """
        sc = self._sensors[0].spacecraft
        return _revisit_pointwise(
            self._aoi.lat_rad, self._aoi.lon_rad,
            sc.keplerian_params,
            t_start, t_end,
            **self._call_kwargs(t_start, alt, max_step, batch_size),
        )
