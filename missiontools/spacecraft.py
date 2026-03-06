from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .orbit.constants import EARTH_MU, EARTH_J2, EARTH_SEMI_MAJOR_AXIS
from .orbit.propagation import (propagate_analytical, sun_synchronous_orbit,
                                 geostationary_orbit, highly_elliptical_orbit)
from .attitude import AttitudeLaw

_VALID_PROPAGATORS = frozenset({'twobody', 'j2'})


@dataclass
class Spacecraft:
    """A spacecraft defined by its Keplerian orbital elements and propagator.

    All angles in radians; distances in metres; times as ``datetime64[us]``.

    Parameters
    ----------
    a : float
        Semi-major axis (m).
    e : float
        Eccentricity (dimensionless).
    i : float
        Inclination (rad).
    raan : float
        Right ascension of the ascending node (rad).
    arg_p : float
        Argument of perigee (rad).
    ma : float
        Mean anomaly at epoch (rad).
    epoch : np.datetime64
        Epoch at which the elements are defined.
    propagator_type : str, optional
        ``'twobody'`` (default) or ``'j2'``.
    central_body_mu : float, optional
        Gravitational parameter (m³ s⁻²).  Defaults to Earth.
    central_body_j2 : float, optional
        J2 perturbation coefficient (m⁵ s⁻²).  Defaults to Earth.
    central_body_radius : float, optional
        Equatorial radius (m).  Defaults to Earth WGS84.

    Examples
    --------
    Construct directly::

        import numpy as np
        from missiontools import Spacecraft

        sc = Spacecraft(
            a=6_771_000.0,
            e=0.0006,
            i=np.radians(51.6),
            raan=np.radians(120.0),
            arg_p=np.radians(30.0),
            ma=0.0,
            epoch=np.datetime64('2025-01-01T00:00:00', 'us'),
            propagator_type='j2',
        )

    Construct from :func:`~missiontools.orbit.sun_synchronous_orbit`::

        from missiontools import Spacecraft
        from missiontools.orbit import sun_synchronous_orbit

        params = sun_synchronous_orbit(altitude=550_000.0, local_time_at_node='10:30')
        sc = Spacecraft.from_dict(params, propagator_type='j2')
    """

    a:                   float
    e:                   float
    i:                   float
    raan:                float
    arg_p:               float
    ma:                  float
    epoch:               np.datetime64
    propagator_type:     str   = 'twobody'
    central_body_mu:     float = EARTH_MU
    central_body_j2:     float = EARTH_J2
    central_body_radius: float = EARTH_SEMI_MAJOR_AXIS

    def __post_init__(self):
        if self.propagator_type not in _VALID_PROPAGATORS:
            raise ValueError(
                f"propagator_type must be one of {sorted(_VALID_PROPAGATORS)}, "
                f"got {self.propagator_type!r}"
            )
        self.epoch = np.asarray(self.epoch, dtype='datetime64[us]').item()
        self._attitude_law: AttitudeLaw = AttitudeLaw.nadir()
        self._sensors: list = []
        self._solar_configs: list = []
        self._thermal_configs: list = []
        self._antennas: list = []

    @property
    def attitude_law(self) -> AttitudeLaw:
        """Pointing law for this spacecraft.  Defaults to nadir pointing."""
        return self._attitude_law

    @attitude_law.setter
    def attitude_law(self, value: AttitudeLaw) -> None:
        if not isinstance(value, AttitudeLaw):
            raise TypeError(
                f"attitude_law must be an AttitudeLaw instance, "
                f"got {type(value).__name__!r}"
            )
        self._attitude_law = value

    @property
    def sensors(self) -> list:
        """Sensors attached to this spacecraft (read-only copy)."""
        return list(self._sensors)

    @property
    def solar_configs(self) -> list:
        """Solar configs attached to this spacecraft (read-only copy)."""
        return list(self._solar_configs)

    def add_solar_config(self, config) -> None:
        """Attach a solar config to this spacecraft.

        Sets the config's back-reference to this spacecraft and appends it to
        the internal solar configs list.

        Parameters
        ----------
        config : AbstractSolarConfig
            The solar config to attach.

        Raises
        ------
        TypeError
            If ``config`` is not an :class:`~missiontools.power.AbstractSolarConfig` instance.
        """
        from .power.solar_config import AbstractSolarConfig  # local import avoids circular dep
        if not isinstance(config, AbstractSolarConfig):
            raise TypeError(
                f"config must be an AbstractSolarConfig instance, "
                f"got {type(config).__name__!r}"
            )
        config._spacecraft = self
        self._solar_configs.append(config)

    @property
    def thermal_configs(self) -> list:
        """Thermal configs attached to this spacecraft (read-only copy)."""
        return list(self._thermal_configs)

    def add_thermal_config(self, config) -> None:
        """Attach a thermal config to this spacecraft.

        Sets the config's back-reference to this spacecraft and appends it to
        the internal thermal configs list.

        Parameters
        ----------
        config : AbstractThermalConfig
            The thermal config to attach.

        Raises
        ------
        TypeError
            If ``config`` is not an
            :class:`~missiontools.thermal.AbstractThermalConfig` instance.
        """
        from .thermal.thermal_config import AbstractThermalConfig
        if not isinstance(config, AbstractThermalConfig):
            raise TypeError(
                f"config must be an AbstractThermalConfig instance, "
                f"got {type(config).__name__!r}"
            )
        config._spacecraft = self
        self._thermal_configs.append(config)

    @property
    def antennas(self) -> list:
        """Antennas attached to this spacecraft (read-only copy)."""
        return list(self._antennas)

    def add_antenna(self, antenna) -> None:
        """Attach an antenna to this spacecraft.

        Sets the antenna's back-reference to this spacecraft and appends it
        to the internal antennas list.

        Parameters
        ----------
        antenna : AbstractAntenna
            The antenna to attach.

        Raises
        ------
        TypeError
            If *antenna* is not an :class:`~missiontools.comm.AbstractAntenna`.
        ValueError
            If the antenna is already attached to a GroundStation.
        """
        from .comm.antenna import AbstractAntenna
        if not isinstance(antenna, AbstractAntenna):
            raise TypeError(
                f"antenna must be an AbstractAntenna instance, "
                f"got {type(antenna).__name__!r}"
            )
        if antenna._ground_station is not None:
            raise ValueError(
                "Antenna is already attached to a GroundStation."
            )
        antenna._spacecraft = self
        self._antennas.append(antenna)

    def add_sensor(self, sensor) -> None:
        """Attach a Sensor to this spacecraft.

        Sets the sensor's back-reference to this spacecraft and appends it to
        the internal sensors list.

        Parameters
        ----------
        sensor : Sensor
            The sensor to attach.

        Raises
        ------
        TypeError
            If ``sensor`` is not a :class:`~missiontools.Sensor` instance.
        """
        from .sensor import Sensor  # local import avoids circular dep
        if not isinstance(sensor, Sensor):
            raise TypeError(
                f"sensor must be a Sensor instance, "
                f"got {type(sensor).__name__!r}"
            )
        sensor._spacecraft = self
        self._sensors.append(sensor)

    @property
    def keplerian_params(self) -> dict:
        """Orbital elements as a dict, compatible with :func:`~missiontools.orbit.propagate_analytical`.

        Use this to pass the spacecraft's orbit to the functional API::

            r, v = propagate_analytical(t, **sc.keplerian_params)
            coverage_fraction(lat, lon, sc.keplerian_params, t_start, t_end,
                              propagator_type=sc.propagator_type)
        """
        return {
            'epoch':               self.epoch,
            'a':                   self.a,
            'e':                   self.e,
            'i':                   self.i,
            'raan':                self.raan,
            'arg_p':               self.arg_p,
            'ma':                  self.ma,
            'central_body_mu':     self.central_body_mu,
            'central_body_j2':     self.central_body_j2,
            'central_body_radius': self.central_body_radius,
        }

    @classmethod
    def from_dict(cls, params: dict,
                  propagator_type: str = 'twobody') -> Spacecraft:
        """Construct from a ``keplerian_params`` dict.

        Accepts dicts produced by :func:`~missiontools.orbit.sun_synchronous_orbit`
        and similar helpers.  Optional central-body keys fall back to Earth
        defaults if absent.

        Parameters
        ----------
        params : dict
            Must contain ``'a'``, ``'e'``, ``'i'``, ``'raan'``, ``'arg_p'``,
            ``'ma'``, ``'epoch'``.  May optionally contain
            ``'central_body_mu'``, ``'central_body_j2'``,
            ``'central_body_radius'``.
        propagator_type : str, optional
            ``'twobody'`` (default) or ``'j2'``.
        """
        return cls(
            a                   = params['a'],
            e                   = params['e'],
            i                   = params['i'],
            raan                = params['raan'],
            arg_p               = params['arg_p'],
            ma                  = params['ma'],
            epoch               = params['epoch'],
            propagator_type     = propagator_type,
            central_body_mu     = params.get('central_body_mu',     EARTH_MU),
            central_body_j2     = params.get('central_body_j2',     EARTH_J2),
            central_body_radius = params.get('central_body_radius', EARTH_SEMI_MAJOR_AXIS),
        )

    def propagate(
            self,
            t_start: np.datetime64,
            t_end:   np.datetime64,
            step:    np.timedelta64,
    ) -> dict:
        """Propagate the orbit and return ECI state vectors.

        Parameters
        ----------
        t_start : np.datetime64
            Start of the propagation window.
        t_end : np.datetime64
            End of the propagation window (inclusive).
        step : np.timedelta64
            Time step between samples.

        Returns
        -------
        dict
            ``t`` : ``(N,)`` ``datetime64[us]`` — sample timestamps.

            ``r`` : ``(N, 3)`` float — ECI position vectors (m).

            ``v`` : ``(N, 3)`` float — ECI velocity vectors (m s⁻¹).
        """
        t_start  = np.asarray(t_start, dtype='datetime64[us]')
        t_end    = np.asarray(t_end,   dtype='datetime64[us]')
        total_us = int((t_end - t_start) / np.timedelta64(1, 'us'))
        step_us  = int(step / np.timedelta64(1, 'us'))

        if total_us <= 0 or step_us <= 0:
            return {
                't': np.array([], dtype='datetime64[us]'),
                'r': np.empty((0, 3), dtype=np.float64),
                'v': np.empty((0, 3), dtype=np.float64),
            }

        offs = np.arange(0, total_us + 1, step_us, dtype=np.int64)
        if offs[-1] != total_us:
            offs = np.append(offs, np.int64(total_us))

        t    = t_start + offs.astype('timedelta64[us]')
        r, v = propagate_analytical(t, **self.keplerian_params,
                                    type=self.propagator_type)
        return {'t': t, 'r': r, 'v': v}

    # ------------------------------------------------------------------
    # Orbit-type factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def sunsync(
            cls,
            altitude_km:     float,
            node_solar_time: str,
            node_type:       str            = 'ascending',
            epoch:           np.datetime64 | None = None,
            ma_deg:          float          = 0.0,
    ) -> 'Spacecraft':
        """Create a circular sun-synchronous orbit spacecraft.

        Delegates to :func:`~missiontools.orbit.sun_synchronous_orbit` for
        element computation and always uses the ``'j2'`` propagator, since J2
        is what drives the RAAN precession that maintains sun-synchronicity.

        Parameters
        ----------
        altitude_km : float
            Orbit altitude above the WGS84 equatorial surface (km).
        node_solar_time : str
            Local solar time at the node crossing (``'HH:MM'`` or
            ``'HH:MM:SS'``, 24-hour clock).
        node_type : str, optional
            ``'ascending'`` (default) or ``'descending'``.
        epoch : np.datetime64 | None, optional
            Reference epoch.  Defaults to J2000.0.
        ma_deg : float, optional
            Mean anomaly at epoch (deg).  Defaults to 0 (spacecraft at the
            ascending or descending node at epoch).

        Returns
        -------
        Spacecraft
            With ``propagator_type='j2'``.
        """
        params = sun_synchronous_orbit(
            altitude            = altitude_km * 1000.0,
            local_time_at_node  = node_solar_time,
            node_type           = node_type,
            epoch               = epoch,
        )
        params['ma'] = np.radians(ma_deg)
        return cls.from_dict(params, propagator_type='j2')

    @classmethod
    def geostationary(
            cls,
            longitude_deg: float,
            epoch:         np.datetime64 | None = None,
            propagator:    str                  = 'twobody',
    ) -> 'Spacecraft':
        """Create a geostationary orbit spacecraft.

        Delegates to :func:`~missiontools.orbit.geostationary_orbit`. The
        satellite is placed at ``longitude_deg`` geographic longitude exactly
        at the epoch.

        Parameters
        ----------
        longitude_deg : float
            Sub-satellite longitude at epoch (deg).  Any value is accepted;
            values outside ``[-180, 180]`` are wrapped automatically.
        epoch : np.datetime64 | None, optional
            Reference epoch.  Defaults to J2000.0.
        propagator : str, optional
            ``'twobody'`` (default) or ``'j2'``.

        Returns
        -------
        Spacecraft
            Equatorial, circular orbit with ``i=0``, ``e=0``.
        """
        return cls.from_dict(
            geostationary_orbit(longitude_deg, epoch=epoch),
            propagator_type=propagator,
        )

    @classmethod
    def heo(
            cls,
            period_s:             float,
            e:                    float,
            epoch:                np.datetime64,
            apogee_solar_time:    str,
            apogee_longitude_deg: float,
            arg_p_deg:            float = 270.0,
            propagator:           str   = 'twobody',
    ) -> 'Spacecraft':
        """Create a critically inclined highly elliptical orbit spacecraft.

        Delegates to :func:`~missiontools.orbit.highly_elliptical_orbit`. The
        inclination is set automatically to the critical inclination (63.435°
        for northern-hemisphere apogee, 116.565° for southern) so the apsidal
        line does not drift under J2.

        Parameters
        ----------
        period_s : float
            Orbital period (s).
        e : float
            Eccentricity (0 < e < 1).
        epoch : np.datetime64
            Reference epoch for the orbital elements.
        apogee_solar_time : str
            Local mean solar time at the apogee sub-satellite point
            (``'HH:MM'`` or ``'HH:MM:SS'``, 24-hour clock).
        apogee_longitude_deg : float
            Geographic longitude of the apogee sub-satellite point (deg).
        arg_p_deg : float, optional
            Argument of perigee (deg).  270° (default) places the apogee in
            the northern hemisphere; 90° places it in the southern hemisphere.
        propagator : str, optional
            ``'twobody'`` (default) or ``'j2'``.

        Returns
        -------
        Spacecraft
        """
        return cls.from_dict(
            highly_elliptical_orbit(
                period_s             = period_s,
                e                    = e,
                epoch                = epoch,
                apogee_solar_time    = apogee_solar_time,
                apogee_longitude_deg = apogee_longitude_deg,
                arg_p_deg            = arg_p_deg,
            ),
            propagator_type=propagator,
        )
