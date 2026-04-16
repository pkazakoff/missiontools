"""
missiontools.sensor
===================
Sensor class for instruments attached to a spacecraft.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .orbit.frames import eci_to_lvlh, eci_to_ecef


def _euler_zyx_to_boresight(
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
) -> npt.NDArray[np.floating]:
    """Sensor boresight in spacecraft body frame from ZYX Euler angles.

    The ZYX intrinsic rotation sequence (yaw → pitch → roll) defines the
    rotation from the spacecraft body frame to the sensor frame:
    ``R = Rx(roll) @ Ry(pitch) @ Rz(yaw)``.

    The sensor boresight (sensor frame z-axis = ``[0, 0, 1]``) expressed
    in the spacecraft body frame is ``R.T @ [0, 0, 1]``.

    Parameters
    ----------
    yaw_deg : float
        Yaw angle (deg), rotation about body-Z.
    pitch_deg : float
        Pitch angle (deg), rotation about new Y after yaw.
    roll_deg : float
        Roll angle (deg), rotation about new X after pitch.

    Returns
    -------
    npt.NDArray[np.floating], shape (3,)
        Unit boresight vector in spacecraft body frame.
    """
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)
    Rz = np.array([[cy, -sy, 0.], [sy,  cy, 0.], [0., 0., 1.]])
    Ry = np.array([[cp,  0., sp], [0.,  1., 0.], [-sp, 0., cp]])
    Rx = np.array([[1.,  0., 0.], [0.,  cr, -sr], [0., sr, cr]])
    R  = Rx @ Ry @ Rz                         # body → sensor
    return R.T @ np.array([0., 0., 1.])       # sensor-z in body frame


class Sensor:
    """An instrument attached to a spacecraft with a cone-shaped field of view.

    Prefer the keyword arguments to select the pointing mode (see below).
    The constructor is public and may be called directly when needed.

    Parameters
    ----------
    half_angle_deg : float
        Half-angle of the sensor's conical field of view (degrees).
        Must satisfy ``0 < half_angle_deg <= 90``.
    attitude_law : AbstractAttitudeLaw, optional
        Independent :class:`~missiontools.AbstractAttitudeLaw` for this sensor,
        decoupled from the host spacecraft's attitude.  Mutually exclusive
        with ``body_vector`` and ``body_euler_deg``.
    body_vector : array_like, shape (3,), optional
        Boresight direction expressed in the **spacecraft body frame**.
        Normalised on input.  Mutually exclusive with ``attitude_law`` and
        ``body_euler_deg``.
    body_euler_deg : (yaw, pitch, roll) tuple of float, optional
        ZYX intrinsic Euler angles (degrees) defining the sensor frame
        relative to the spacecraft body frame.  The boresight is the
        sensor frame's z-axis expressed in body-frame coordinates.
        Mutually exclusive with ``attitude_law`` and ``body_vector``.

    Notes
    -----
    Body-mounted sensors (``body_vector`` or ``body_euler_deg``) require the
    sensor to be attached to a spacecraft via
    :meth:`~missiontools.Spacecraft.add_sensor` before their pointing methods
    can be called.

    Examples
    --------
    Nadir-pointing sensor, 10° half-angle::

        from missiontools import Sensor, FixedAttitudeLaw
        sensor = Sensor(10.0, attitude_law=FixedAttitudeLaw.nadir())

    Sensor body-mounted along spacecraft body-z (boresight = nadir for a
    nadir spacecraft), 5° half-angle::

        sensor = Sensor(5.0, body_vector=[0, 0, 1])

    Sensor tilted 30° in pitch from body-z::

        sensor = Sensor(15.0, body_euler_deg=(0, 30, 0))
    """

    def __init__(
            self,
            half_angle_deg: float,
            *,
            attitude_law=None,
            body_vector: npt.ArrayLike | None = None,
            body_euler_deg: tuple[float, float, float] | None = None,
    ):
        # --- validate half-angle -------------------------------------------
        half_angle_deg = float(half_angle_deg)
        if not (0.0 < half_angle_deg <= 90.0):
            raise ValueError(
                f"half_angle_deg must be in (0, 90], got {half_angle_deg}"
            )
        self._half_angle_rad: float = np.radians(half_angle_deg)

        # --- validate mode (exactly one) ------------------------------------
        n_modes = sum(x is not None for x in
                      (attitude_law, body_vector, body_euler_deg))
        if n_modes == 0:
            raise ValueError(
                "Exactly one of 'attitude_law', 'body_vector', or "
                "'body_euler_deg' must be provided."
            )
        if n_modes > 1:
            raise ValueError(
                "Only one of 'attitude_law', 'body_vector', or "
                "'body_euler_deg' may be provided."
            )

        # --- store mode-specific state --------------------------------------
        self._spacecraft = None   # set by Spacecraft.add_sensor

        if attitude_law is not None:
            from .attitude import AbstractAttitudeLaw
            if not isinstance(attitude_law, AbstractAttitudeLaw):
                raise TypeError(
                    f"attitude_law must be an AbstractAttitudeLaw instance, "
                    f"got {type(attitude_law).__name__!r}"
                )
            self._mode         = 'independent'
            self._attitude_law = attitude_law
            self._body_vector  = None

        elif body_vector is not None:
            vec = np.asarray(body_vector, dtype=np.float64)
            if vec.shape != (3,):
                raise ValueError(
                    f"body_vector must have shape (3,), got {vec.shape}"
                )
            norm = np.linalg.norm(vec)
            if norm == 0.0:
                raise ValueError("body_vector must not be the zero vector")
            self._mode         = 'body'
            self._attitude_law = None
            self._body_vector  = vec / norm

        else:  # body_euler_deg
            yaw, pitch, roll = (float(a) for a in body_euler_deg)
            boresight = _euler_zyx_to_boresight(yaw, pitch, roll)
            self._mode         = 'body'
            self._attitude_law = None
            self._body_vector  = boresight / np.linalg.norm(boresight)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def half_angle_rad(self) -> float:
        """FOV cone half-angle in radians."""
        return self._half_angle_rad

    @property
    def half_angle_deg(self) -> float:
        """FOV cone half-angle in degrees."""
        return float(np.degrees(self._half_angle_rad))

    @property
    def spacecraft(self):
        """Host spacecraft, or ``None`` if not yet attached."""
        return self._spacecraft

    # ------------------------------------------------------------------
    # Pointing methods
    # ------------------------------------------------------------------

    def pointing_eci(self,
                     r_eci: npt.ArrayLike,
                     v_eci: npt.ArrayLike,
                     t: npt.ArrayLike,
                     ) -> npt.NDArray[np.floating]:
        """Boresight unit vector(s) in the ECI frame.

        For ``body_vector`` / ``body_euler_deg`` sensors the host spacecraft's
        :attr:`~missiontools.Spacecraft.attitude_law` is used to transform
        the body-frame boresight to ECI.

        Parameters
        ----------
        r_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI position(s) (m).
        v_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI velocity(s) (m s⁻¹).
        t : array_like of datetime64, shape ``(N,)`` or scalar
            Observation epoch(s).

        Returns
        -------
        npt.NDArray[np.floating]
            Unit boresight vector(s) in ECI, shape ``(N, 3)`` for array
            inputs or ``(3,)`` for scalar inputs.

        Raises
        ------
        RuntimeError
            If the sensor is in body mode and has not been attached to a
            spacecraft via :meth:`~missiontools.Spacecraft.add_sensor`.
        """
        if self._mode == 'independent':
            return self._attitude_law.pointing_eci(r_eci, v_eci, t)

        # body mode
        if self._spacecraft is None:
            raise RuntimeError(
                "Sensor must be attached to a Spacecraft via add_sensor() "
                "before pointing methods can be called in body mode."
            )
        return self._spacecraft.attitude_law.rotate_from_body(
            self._body_vector, r_eci, v_eci, t,
        )

    def pointing_lvlh(self,
                      r_eci: npt.ArrayLike,
                      v_eci: npt.ArrayLike,
                      t: npt.ArrayLike,
                      ) -> npt.NDArray[np.floating]:
        """Boresight unit vector(s) in the LVLH frame.

        Parameters
        ----------
        r_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI position(s) (m).
        v_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI velocity(s) (m s⁻¹).
        t : array_like of datetime64, shape ``(N,)`` or scalar
            Observation epoch(s).

        Returns
        -------
        npt.NDArray[np.floating]
            Unit boresight vector(s) in LVLH, shape ``(N, 3)`` for array
            inputs or ``(3,)`` for scalar inputs.
        """
        if self._mode == 'independent':
            return self._attitude_law.pointing_lvlh(r_eci, v_eci, t)

        r     = np.asarray(r_eci, dtype=np.float64)
        scalar = r.ndim == 1
        r_2d  = np.atleast_2d(r)
        v_2d  = np.atleast_2d(np.asarray(v_eci, dtype=np.float64))
        eci   = np.atleast_2d(self.pointing_eci(r_eci, v_eci, t))
        result = eci_to_lvlh(eci, r_2d, v_2d)
        return result[0] if scalar else result

    def pointing_ecef(self,
                      r_eci: npt.ArrayLike,
                      v_eci: npt.ArrayLike,
                      t: npt.ArrayLike,
                      ) -> npt.NDArray[np.floating]:
        """Boresight unit vector(s) in the ECEF frame.

        Parameters
        ----------
        r_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI position(s) (m).
        v_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI velocity(s) (m s⁻¹).
        t : array_like of datetime64, shape ``(N,)`` or scalar
            Observation epoch(s).

        Returns
        -------
        npt.NDArray[np.floating]
            Unit boresight vector(s) in ECEF, shape ``(N, 3)`` for array
            inputs or ``(3,)`` for scalar inputs.
        """
        if self._mode == 'independent':
            return self._attitude_law.pointing_ecef(r_eci, v_eci, t)

        r     = np.asarray(r_eci, dtype=np.float64)
        scalar = r.ndim == 1
        t_arr = np.atleast_1d(np.asarray(t, dtype='datetime64[us]'))
        eci   = np.atleast_2d(self.pointing_eci(r_eci, v_eci, t))
        result = eci_to_ecef(eci, t_arr)
        return result[0] if scalar else result
