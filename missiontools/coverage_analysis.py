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
    make_sensor_spec,
    coverage_fraction_multi,
    pointwise_coverage_multi,
    collect_access_intervals_multi,
)

_DEFAULT_STEP = np.timedelta64(30, 's')


class Coverage:
    """Coverage analysis for an area of interest observed by one or more sensors.

    Each sensor must be attached to a :class:`~missiontools.Spacecraft` via
    :meth:`~missiontools.Spacecraft.add_sensor` before constructing a
    ``Coverage`` object.  The spacecraft orbit and propagator are derived
    automatically from each sensor's back-reference.

    Combined visibility at each timestep is the **union** of all sensor FOVs.
    The SZA constraint (if any) is applied globally after the union — it
    represents an illumination or link condition on the ground, not on any
    individual sensor.

    Parameters
    ----------
    aoi : AoI
        Area of interest containing ground sample points.
    sensors : sequence of Sensor
        One or more :class:`~missiontools.Sensor` instances (list, tuple, or
        any iterable).  All sensors must be attached to a spacecraft.  Sensors
        may belong to the same spacecraft (multi-FOV) or to different
        spacecraft (constellation).
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
    All five coverage methods evaluate each sensor's LVLH boresight at
    *t_start* and assume it is constant in the LVLH frame for the full
    analysis window.  This is exact for sensors with fixed-LVLH pointing
    (nadir, fixed tilt, body-mounted on a nadir spacecraft) and an
    approximation for time-varying pointing modes.

    Examples
    --------
    Single sensor::

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

    Constellation (two spacecraft at different RAANs)::

        sc2    = Spacecraft(a=6_771_000., e=0., i=np.radians(51.6),
                            raan=np.radians(90.), arg_p=0., ma=0.,
                            epoch=np.datetime64('2025-01-01', 'us'))
        s2     = Sensor(30.0, body_vector=[0, 0, 1])
        sc2.add_sensor(s2)

        cov2 = Coverage(aoi, [sensor, s2], el_min_deg=5.0)
        result2 = cov2.coverage_fraction(
            np.datetime64('2025-01-01', 'us'),
            np.datetime64('2025-01-02', 'us'),
        )
        print(result2['final_cumulative'])   # ≥ single-satellite result
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
        sensors = list(sensors)
        if len(sensors) == 0:
            raise ValueError("sensors must be a non-empty sequence of Sensor objects")
        for s in sensors:
            if not isinstance(s, Sensor):
                raise TypeError(
                    f"Each element of sensors must be a Sensor instance, "
                    f"got {type(s).__name__!r}"
                )
            if s.spacecraft is None:
                raise RuntimeError(
                    "Every sensor must be attached to a Spacecraft via "
                    "add_sensor() before use with Coverage."
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

    def _all_sensor_specs(self, t_start: np.datetime64) -> list:
        """Return a sensor-spec tuple for every sensor, evaluated at *t_start*.

        Each spec encodes the spacecraft keplerian parameters, propagator type,
        and the frozen LVLH boresight direction assumed constant over the
        analysis window.
        """
        specs = []
        for sensor in self._sensors:
            sc    = sensor.spacecraft
            state = sc.propagate(
                t_start,
                t_start + np.timedelta64(1, 's'),
                np.timedelta64(1, 's'),
            )
            r, v, t = state['r'][0], state['v'][0], state['t'][0]
            specs.append(make_sensor_spec(
                sc.keplerian_params,
                sc.propagator_type,
                sensor.pointing_lvlh(r, v, t),
                sensor.half_angle_rad,
            ))
        return specs

    def _sza_rad(self) -> tuple[float | None, float | None]:
        """Return ``(sza_max, sza_min)`` in radians (or ``None``)."""
        sza_max = (np.radians(self._sza_max_deg)
                   if self._sza_max_deg is not None else None)
        sza_min = (np.radians(self._sza_min_deg)
                   if self._sza_min_deg is not None else None)
        return sza_max, sza_min

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
        sza_max, sza_min = self._sza_rad()
        return coverage_fraction_multi(
            self._aoi.lat_rad, self._aoi.lon_rad,
            self._all_sensor_specs(t_start),
            t_start, t_end,
            alt, np.radians(self._el_min_deg),
            sza_max, sza_min,
            max_step, batch_size,
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
            ``'max_revisit'`` : ``(M,)`` float — maximum gap between
            consecutive accesses for each ground point (s).  ``nan`` for
            points with fewer than two accesses.

            ``'mean_revisit'`` : ``(M,)`` float — mean gap (s).  ``nan``
            for points with fewer than two accesses.

            ``'global_max'`` : float — maximum value across all points (s).
            ``nan`` if no point has two or more accesses.

            ``'global_mean'`` : float — mean of per-point mean revisit
            times (s).  ``nan`` if no point has two or more accesses.
        """
        sza_max, sza_min = self._sza_rad()
        lat = self._aoi.lat_rad
        lon = self._aoi.lon_rad
        M   = len(lat)

        intervals = collect_access_intervals_multi(
            lat, lon,
            self._all_sensor_specs(t_start),
            t_start, t_end,
            alt, np.radians(self._el_min_deg),
            sza_max, sza_min,
            max_step, batch_size,
            close_at_end=False,
        )

        max_revisit  = np.full(M, np.nan)
        mean_revisit = np.full(M, np.nan)
        for m, ivals in enumerate(intervals):
            if len(ivals) >= 2:
                gaps_us = np.array([
                    int((ivals[i + 1][0] - ivals[i][1]) / np.timedelta64(1, 'us'))
                    for i in range(len(ivals) - 1)
                ], dtype=np.float64)
                max_revisit[m]  = gaps_us.max()  / 1e6
                mean_revisit[m] = gaps_us.mean() / 1e6

        has_gaps = ~np.isnan(max_revisit)
        return {
            'max_revisit':  max_revisit,
            'mean_revisit': mean_revisit,
            'global_max':   float(np.nanmax(max_revisit))   if has_gaps.any() else float('nan'),
            'global_mean':  float(np.nanmean(mean_revisit)) if has_gaps.any() else float('nan'),
        }

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
        sza_max, sza_min = self._sza_rad()
        return pointwise_coverage_multi(
            self._aoi.lat_rad, self._aoi.lon_rad,
            self._all_sensor_specs(t_start),
            t_start, t_end,
            alt, np.radians(self._el_min_deg),
            sza_max, sza_min,
            max_step, batch_size,
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
        sza_max, sza_min = self._sza_rad()
        return collect_access_intervals_multi(
            self._aoi.lat_rad, self._aoi.lon_rad,
            self._all_sensor_specs(t_start),
            t_start, t_end,
            alt, np.radians(self._el_min_deg),
            sza_max, sza_min,
            max_step, batch_size,
            close_at_end=True,
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
        sza_max, sza_min = self._sza_rad()
        intervals = collect_access_intervals_multi(
            self._aoi.lat_rad, self._aoi.lon_rad,
            self._all_sensor_specs(t_start),
            t_start, t_end,
            alt, np.radians(self._el_min_deg),
            sza_max, sza_min,
            max_step, batch_size,
            close_at_end=False,
        )

        result = []
        for ivals in intervals:
            if len(ivals) >= 2:
                gaps = np.array([
                    ivals[i + 1][0] - ivals[i][1]
                    for i in range(len(ivals) - 1)
                ], dtype='timedelta64[us]')
            else:
                gaps = np.array([], dtype='timedelta64[us]')
            result.append(gaps)
        return result
