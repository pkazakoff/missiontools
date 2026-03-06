"""Antenna classes for spacecraft and ground station link analysis.

An antenna can be attached to a :class:`~missiontools.Spacecraft` via
:meth:`~missiontools.Spacecraft.add_antenna` or to a
:class:`~missiontools.GroundStation` via
:meth:`~missiontools.GroundStation.add_antenna`.

The :meth:`~AbstractAntenna.gain` method computes the antenna gain (dBi)
for given direction vectors in a specified reference frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from ..orbit.frames import ecef_to_eci, lvlh_to_eci, azel_to_enu, enu_to_ecef
from ..sensor import _euler_zyx_to_boresight


class AbstractAntenna(ABC):
    """Base class for antennas attachable to Spacecraft or GroundStation.

    Mounting is specified via keyword arguments.  Exactly one mounting
    group must be provided (spacecraft or ground station), unless the
    subclass is direction-independent (e.g. :class:`IsotropicAntenna`).

    **Spacecraft mounting** (provide exactly one):

    - ``attitude_law`` — independent :class:`~missiontools.AttitudeLaw`
    - ``body_vector`` — boresight direction in the spacecraft body frame
    - ``body_euler_deg`` — ``(yaw, pitch, roll)`` ZYX intrinsic Euler
      angles defining the boresight in the body frame

    **Ground station mounting**:

    - ``azimuth_deg`` — azimuth from north (deg), clockwise positive
    - ``elevation_deg`` — elevation from horizon (deg)
    - ``rotation_deg`` — boresight rotation (deg), default 0

    Parameters
    ----------
    attitude_law : AttitudeLaw, optional
    body_vector : array_like, shape (3,), optional
    body_euler_deg : tuple of float, optional
    azimuth_deg : float, optional
    elevation_deg : float, optional
    rotation_deg : float, optional
    """

    def __init__(
        self,
        *,
        attitude_law=None,
        body_vector: npt.ArrayLike | None = None,
        body_euler_deg: tuple[float, float, float] | None = None,
        azimuth_deg: float | None = None,
        elevation_deg: float | None = None,
        rotation_deg: float = 0.0,
    ) -> None:
        sc_opts = sum(x is not None for x in
                      (attitude_law, body_vector, body_euler_deg))
        gs_opts = azimuth_deg is not None

        if sc_opts and gs_opts:
            raise ValueError(
                "Cannot mix spacecraft mounting (attitude_law / body_vector "
                "/ body_euler_deg) with ground station mounting "
                "(azimuth_deg / elevation_deg)."
            )

        if not sc_opts and not gs_opts:
            raise ValueError(
                "Must specify spacecraft mounting (attitude_law, "
                "body_vector, or body_euler_deg) or ground station "
                "mounting (azimuth_deg + elevation_deg)."
            )

        if sc_opts > 1:
            raise ValueError(
                "Specify exactly one of attitude_law, body_vector, "
                "or body_euler_deg."
            )

        self._spacecraft = None
        self._ground_station = None

        if gs_opts:
            # Ground station mounting
            if elevation_deg is None:
                raise ValueError(
                    "elevation_deg is required for ground station mounting."
                )
            self._mode = 'ground'
            az_rad = np.radians(float(azimuth_deg))
            el_rad = np.radians(float(elevation_deg))
            self._boresight_enu = azel_to_enu(az_rad, el_rad)
            self._rotation_deg = float(rotation_deg)
            self._boresight_ecef = None  # set at attachment time
        elif attitude_law is not None:
            self._mode = 'independent'
            self._attitude_law = attitude_law
        else:
            # Body-mounted
            self._mode = 'body'
            if body_vector is not None:
                bv = np.asarray(body_vector, dtype=np.float64)
                if bv.shape != (3,):
                    raise ValueError(
                        f"body_vector must have shape (3,), got {bv.shape}"
                    )
                norm = np.linalg.norm(bv)
                if norm == 0:
                    raise ValueError("body_vector must be non-zero.")
                self._body_vector = bv / norm
            else:
                yaw, pitch, roll = body_euler_deg
                self._body_vector = _euler_zyx_to_boresight(yaw, pitch, roll)

    # --- properties ---

    @property
    def host(self):
        """The host object (Spacecraft or GroundStation), or ``None``."""
        return self._spacecraft or self._ground_station

    @property
    def spacecraft(self):
        """Spacecraft this antenna is attached to, or ``None``."""
        return self._spacecraft

    @property
    def ground_station(self):
        """GroundStation this antenna is attached to, or ``None``."""
        return self._ground_station

    # --- boresight computation ---

    def boresight_eci(
        self,
        r_eci: npt.ArrayLike,
        v_eci: npt.ArrayLike,
        t: npt.ArrayLike,
    ) -> npt.NDArray[np.floating]:
        """Compute the antenna boresight direction in ECI.

        Parameters
        ----------
        r_eci : array_like, shape (N, 3) or (3,)
            ECI position (m).  Used for spacecraft-mounted antennas.
        v_eci : array_like, shape (N, 3) or (3,)
            ECI velocity (m/s).
        t : array_like of datetime64[us], shape (N,) or scalar
            Timestamps.

        Returns
        -------
        ndarray, shape (N, 3) or (3,)
            Boresight unit vector in ECI.
        """
        if self._mode == 'independent':
            return self._attitude_law.pointing_eci(r_eci, v_eci, t)
        elif self._mode == 'body':
            if self._spacecraft is None:
                raise RuntimeError(
                    "Body-mounted antenna must be attached to a Spacecraft "
                    "via add_antenna() before computing boresight."
                )
            return self._spacecraft.attitude_law.rotate_from_body(
                self._body_vector, r_eci, v_eci, t,
            )
        elif self._mode == 'ground':
            if self._boresight_ecef is None:
                raise RuntimeError(
                    "Ground-mounted antenna must be attached to a "
                    "GroundStation via add_antenna() before computing "
                    "boresight."
                )
            t_arr = np.atleast_1d(np.asarray(t, dtype='datetime64[us]'))
            n = len(t_arr)
            boresight_tiled = np.tile(self._boresight_ecef, (n, 1))
            result = ecef_to_eci(boresight_tiled, t_arr)
            if np.asarray(t).ndim == 0:
                return result[0]
            return result
        else:
            raise RuntimeError(
                f"Cannot compute boresight for mode '{self._mode}'."
            )

    # --- gain computation ---

    def gain(
        self,
        t: npt.ArrayLike,
        v: npt.ArrayLike,
        frame: str = 'eci',
        *,
        r_eci: npt.ArrayLike | None = None,
        v_eci: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute antenna gain for given direction vectors.

        Parameters
        ----------
        t : array_like of datetime64[us], shape (N,) or scalar
            Timestamps.
        v : array_like, shape (N, 3) or (3,)
            Direction vectors pointing from the antenna toward the
            target.  Need not be unit vectors (normalised internally).
        frame : str
            Reference frame of *v*: ``'eci'``, ``'ecef'``, or
            ``'lvlh'``.
        r_eci : array_like, shape (N, 3) or (3,), optional
            ECI position.  Required for spacecraft-mounted antennas
            and for ``frame='lvlh'``.
        v_eci : array_like, shape (N, 3) or (3,), optional
            ECI velocity.  Required for spacecraft-mounted antennas
            and for ``frame='lvlh'``.

        Returns
        -------
        ndarray, shape (N,)
            Gain in dBi for each direction vector.
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype='datetime64[us]'))
        v_arr = np.atleast_2d(np.asarray(v, dtype=np.float64))
        n = len(v_arr)

        # Convert v to ECI
        if frame == 'eci':
            v_eci_dir = v_arr
        elif frame == 'ecef':
            v_eci_dir = ecef_to_eci(v_arr, t_arr)
        elif frame == 'lvlh':
            if r_eci is None or v_eci is None:
                raise ValueError(
                    "r_eci and v_eci are required for frame='lvlh'."
                )
            v_eci_dir = lvlh_to_eci(v_arr, r_eci, v_eci)
        else:
            raise ValueError(
                f"Unknown frame '{frame}'. Use 'eci', 'ecef', or 'lvlh'."
            )

        # Normalise direction vectors
        norms = np.linalg.norm(v_eci_dir, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        v_hat = v_eci_dir / norms

        # Get boresight in ECI
        boresight = np.atleast_2d(
            self.boresight_eci(r_eci, v_eci, t_arr)
        )

        # Off-boresight angle
        cos_theta = np.einsum('ij,ij->i', boresight, v_hat)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        return self._pattern_gain(theta)

    @property
    def peak_gain_dbi(self) -> float:
        """Peak gain at boresight (dBi)."""
        return float(self._pattern_gain(np.zeros(1))[0])

    @abstractmethod
    def _pattern_gain(
        self,
        off_boresight_rad: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Gain in dBi as a function of off-boresight angle.

        Parameters
        ----------
        off_boresight_rad : ndarray, shape (N,)
            Angle from boresight (rad), in [0, pi].

        Returns
        -------
        ndarray, shape (N,)
            Gain in dBi.
        """


class IsotropicAntenna(AbstractAntenna):
    """Antenna with constant gain in all directions.

    Since the gain is direction-independent, no mounting information is
    needed.

    Parameters
    ----------
    gain_dbi : float
        Constant gain (dBi).  Default 0.0.
    """

    def __init__(self, gain_dbi: float = 0.0) -> None:
        # Skip AbstractAntenna.__init__ — no mounting needed
        self._gain_dbi = float(gain_dbi)
        self._mode = None
        self._spacecraft = None
        self._ground_station = None

    def gain(
        self,
        t: npt.ArrayLike,
        v: npt.ArrayLike,
        frame: str = 'eci',
        *,
        r_eci: npt.ArrayLike | None = None,
        v_eci: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.floating]:
        """Return constant gain regardless of direction.

        Parameters are accepted for interface compatibility but ignored.
        """
        v_arr = np.atleast_2d(np.asarray(v, dtype=np.float64))
        return np.full(v_arr.shape[0], self._gain_dbi)

    @property
    def peak_gain_dbi(self) -> float:
        """Peak gain (dBi), constant for isotropic antenna."""
        return self._gain_dbi

    def _pattern_gain(
        self,
        off_boresight_rad: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        return np.full_like(off_boresight_rad, self._gain_dbi)


class SymmetricAntenna(AbstractAntenna):
    """Axially symmetric antenna defined by a gain-vs-angle table.

    The radiation pattern is symmetric about the boresight.  Gain values
    are linearly interpolated between the tabulated points.

    Parameters
    ----------
    angles_deg : array_like, shape (K,)
        Off-boresight angles (deg).  Must be monotonically increasing
        and span a range within [0, 180].
    gains_dbi : array_like, shape (K,)
        Gain at each angle (dBi).
    **kwargs
        Mounting keyword arguments passed to :class:`AbstractAntenna`.
    """

    def __init__(
        self,
        angles_deg: npt.ArrayLike,
        gains_dbi: npt.ArrayLike,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        angles = np.asarray(angles_deg, dtype=np.float64)
        gains = np.asarray(gains_dbi, dtype=np.float64)

        if angles.ndim != 1:
            raise ValueError(
                f"angles_deg must be 1-D, got shape {angles.shape}"
            )
        if gains.ndim != 1:
            raise ValueError(
                f"gains_dbi must be 1-D, got shape {gains.shape}"
            )
        if len(angles) != len(gains):
            raise ValueError(
                f"angles_deg length ({len(angles)}) must match "
                f"gains_dbi length ({len(gains)})."
            )
        if len(angles) < 2:
            raise ValueError("Need at least 2 angle/gain pairs.")
        if np.any(np.diff(angles) <= 0):
            raise ValueError("angles_deg must be monotonically increasing.")
        if angles[0] < 0 or angles[-1] > 180:
            raise ValueError(
                "angles_deg must be within [0, 180] degrees."
            )

        self._angles_rad = np.radians(angles)
        self._gains_dbi = gains.copy()

    @property
    def angles_deg(self) -> npt.NDArray[np.floating]:
        """Tabulated off-boresight angles (deg)."""
        return np.degrees(self._angles_rad).copy()

    @property
    def gains_dbi(self) -> npt.NDArray[np.floating]:
        """Tabulated gain values (dBi)."""
        return self._gains_dbi.copy()

    def _pattern_gain(
        self,
        off_boresight_rad: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        return np.interp(
            off_boresight_rad, self._angles_rad, self._gains_dbi,
        )
