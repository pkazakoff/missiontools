from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .orbit.access import earth_access_intervals


@dataclass
class GroundStation:
    """A ground station defined in WGS84 geodetic coordinates.

    Parameters
    ----------
    lat : float
        Geodetic latitude (deg), range [-90, 90].
    lon : float
        Longitude (deg).  Any value is accepted; cos/sin periodicity means
        values outside [-180, 180] are equivalent to their wrapped form.
    alt : float, optional
        Altitude above the WGS84 ellipsoid (m).  Defaults to 0 (mean
        sea level approximation).

    Examples
    --------
    Create a ground station and compute access against a spacecraft::

        import numpy as np
        from missiontools import GroundStation, Spacecraft

        gs = GroundStation(lat=51.5, lon=-0.1)           # London
        sc = Spacecraft(a=6_771_000, e=0, i=np.radians(51.6), ...)

        passes = gs.access(sc,
                           t_start = np.datetime64('2025-01-01', 'us'),
                           t_end   = np.datetime64('2025-01-03', 'us'),
                           el_min_deg  = 5.0)
    """

    lat: float
    lon: float
    alt: float = 0.0

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"lat must be in [-90, 90] degrees, got {self.lat}")
        self._antennas: list = []

    @property
    def antennas(self) -> list:
        """Antennas attached to this ground station (read-only copy)."""
        return list(self._antennas)

    def add_antenna(self, antenna) -> None:
        """Attach an antenna to this ground station.

        Sets the antenna's back-reference and pre-computes the ECEF
        boresight for ground-mounted antennas.

        Parameters
        ----------
        antenna : AbstractAntenna
            The antenna to attach.

        Raises
        ------
        TypeError
            If *antenna* is not an
            :class:`~missiontools.comm.AbstractAntenna`.
        ValueError
            If the antenna is already attached to a Spacecraft.
        """
        from .comm.antenna import AbstractAntenna

        if not isinstance(antenna, AbstractAntenna):
            raise TypeError(
                f"antenna must be an AbstractAntenna instance, "
                f"got {type(antenna).__name__!r}"
            )
        if antenna._spacecraft is not None:
            raise ValueError("Antenna is already attached to a Spacecraft.")
        antenna._ground_station = self
        if antenna._mode == "ground":
            from .orbit.frames import enu_to_ecef

            antenna._boresight_ecef = enu_to_ecef(
                antenna._boresight_enu,
                np.radians(self.lat),
                np.radians(self.lon),
            )
        self._antennas.append(antenna)

    def access(
        self,
        spacecraft,
        t_start: np.datetime64,
        t_end: np.datetime64,
        el_min_deg: float = 0.0,
        max_step: np.timedelta64 = np.timedelta64(30, "s"),
    ) -> list[tuple[np.datetime64, np.datetime64]]:
        """Compute access intervals between this ground station and a spacecraft.

        Parameters
        ----------
        spacecraft : Spacecraft
            The spacecraft whose orbit is to be checked.
        t_start : np.datetime64
            Start of the search window.
        t_end : np.datetime64
            End of the search window.
        el_min_deg : float, optional
            Minimum elevation angle (deg).  Contacts below this elevation
            are not reported.  Default 0.
        max_step : np.timedelta64, optional
            Maximum coarse scan step used internally.  Smaller values improve
            detection of very short passes at the cost of runtime.
            Default 30 s.

        Returns
        -------
        list of tuple[np.datetime64, np.datetime64]
            Each tuple ``(start, end)`` is a continuous access window where
            the spacecraft is visible above ``el_min``.  Both timestamps are
            ``datetime64[us]``.  Returns an empty list if no access occurs.
        """
        return earth_access_intervals(
            t_start=t_start,
            t_end=t_end,
            keplerian_params=spacecraft.keplerian_params,
            lat=np.radians(self.lat),
            lon=np.radians(self.lon),
            alt=self.alt,
            el_min=np.radians(el_min_deg),
            propagator_type=spacecraft.propagator_type,
            max_step=max_step,
        )
