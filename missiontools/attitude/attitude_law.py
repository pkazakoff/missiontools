"""
AttitudeLaw
===========
Spacecraft/sensor pointing law with full quaternion internal storage.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..orbit.frames import (eci_to_ecef, eci_to_lvlh,
                             ecef_to_eci, lvlh_to_eci,
                             sun_vec_eci)
from ..orbit.propagation import propagate_analytical

_VALID_FRAMES = frozenset({'lvlh', 'eci', 'ecef'})


# ---------------------------------------------------------------------------
# Private quaternion helpers
# ---------------------------------------------------------------------------

def _q_compose(q1: npt.NDArray, q2: npt.NDArray) -> npt.NDArray:
    """Quaternion product q1 ⊗ q2, both [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _q_from_vec(vec: npt.NDArray, roll: float = 0.0) -> npt.NDArray:
    """Minimum-rotation quaternion aligning body-z to *vec*, then optional roll.

    Parameters
    ----------
    vec : npt.NDArray, shape (3,)
        Target boresight direction (need not be a unit vector).
    roll : float
        Roll angle (rad) about the boresight axis.  Default 0 gives the
        minimum rotation from identity that aligns body-z with *vec*.

    Returns
    -------
    npt.NDArray, shape (4,)
        Unit quaternion [w, x, y, z].
    """
    v = vec / np.linalg.norm(vec)
    dot = v[2]                           # np.dot([0,0,1], v) = v[2]

    if dot < -0.9999:
        # 180° edge case: rotate about x-axis
        q_min = np.array([0., 1., 0., 0.])
    else:
        # Half-angle construction: axis = [0,0,1] × v, w = 1 + cos θ
        axis = np.cross([0., 0., 1.], v)  # = [-v[1], v[0], 0]
        half = np.append(1.0 + dot, axis)
        q_min = half / np.linalg.norm(half)

    if roll == 0.0:
        return q_min

    # Roll = rotation about body-z (boresight), composed BEFORE q_min
    q_roll = np.array([np.cos(roll / 2), 0., 0., np.sin(roll / 2)])
    return _q_compose(q_min, q_roll)


def _q_rotate(q: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
    """Rotate 3D vector v by unit quaternion q  (R(q) @ v)."""
    w, x, y, z = q
    return np.array([
        (1 - 2*(y*y + z*z))*v[0] + 2*(x*y - w*z)*v[1] + 2*(x*z + w*y)*v[2],
            2*(x*y + w*z)*v[0] + (1 - 2*(x*x + z*z))*v[1] + 2*(y*z - w*x)*v[2],
            2*(x*z - w*y)*v[0] +     2*(y*z + w*x)*v[1] + (1 - 2*(x*x + y*y))*v[2],
    ])


def _q_from_vec_batch(vecs: npt.NDArray,
                      roll: float | npt.NDArray = 0.0) -> npt.NDArray:
    """Vectorized :func:`_q_from_vec` over N directions.

    Parameters
    ----------
    vecs : npt.NDArray, shape (N, 3)
        Target boresight directions (must be unit vectors).
    roll : float or ndarray of shape (N,)
        Roll angle(s) (rad) about each boresight axis.  Scalar applies
        the same roll to every quaternion; an array gives per-element rolls.

    Returns
    -------
    npt.NDArray, shape (N, 4)
        Unit quaternions [w, x, y, z].
    """
    dot = vecs[:, 2]               # (N,)  cos θ  = v · ẑ

    # axis = ẑ × v = [-v[1], v[0], 0];  half = [1 + cos θ, axis]
    N    = len(vecs)
    half = np.empty((N, 4), dtype=np.float64)
    half[:, 0] =  1.0 + dot
    half[:, 1] = -vecs[:, 1]
    half[:, 2] =  vecs[:, 0]
    half[:, 3] =  0.0

    # 180° edge case: dot ≈ −1 → rotate about body-x
    mask = dot < -0.9999
    half[mask] = [0., 1., 0., 0.]

    q_min = half / np.linalg.norm(half, axis=1, keepdims=True)

    roll_arr = np.asarray(roll)
    if roll_arr.ndim == 0 and float(roll_arr) == 0.0:
        return q_min

    # Compose q_min[n] ⊗ q_roll  where  q_roll = [cr, 0, 0, sr]
    # Simplified Hamilton product with x2=y2=0:
    cr, sr = np.cos(roll_arr / 2), np.sin(roll_arr / 2)
    w, x, y, z = q_min[:, 0], q_min[:, 1], q_min[:, 2], q_min[:, 3]
    return np.stack([
        w*cr - z*sr,
        x*cr + y*sr,
        y*cr - x*sr,
        w*sr + z*cr,
    ], axis=1)


def _q_rotate_batch(qs: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
    """Rotate a fixed 3-D vector by N quaternions.

    Parameters
    ----------
    qs : npt.NDArray, shape (N, 4)
        Unit quaternions [w, x, y, z].
    v : npt.NDArray, shape (3,)
        Vector to rotate.

    Returns
    -------
    npt.NDArray, shape (N, 3)
        Rotated vectors (not normalised).
    """
    w, x, y, z = qs[:, 0], qs[:, 1], qs[:, 2], qs[:, 3]
    vx, vy, vz = v
    return np.stack([
        (1 - 2*(y*y + z*z))*vx + 2*(x*y - w*z)*vy + 2*(x*z + w*y)*vz,
            2*(x*y + w*z)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - w*x)*vz,
            2*(x*z - w*y)*vx +     2*(y*z + w*x)*vy + (1 - 2*(x*x + y*y))*vz,
    ], axis=1)


def _q_boresight(q: npt.NDArray) -> npt.NDArray:
    """Boresight direction (body-z rotated by quaternion q).

    Returns the third column of the DCM corresponding to *q*.

    Parameters
    ----------
    q : npt.NDArray, shape (4,)
        Unit quaternion [w, x, y, z].

    Returns
    -------
    npt.NDArray, shape (3,)
        Unit pointing vector in the quaternion's reference frame.
    """
    w, x, y, z = q
    return np.array([
        2.0 * (x*z + w*y),
        2.0 * (y*z - w*x),
        w**2 - x**2 - y**2 + z**2,
    ])


# Nadir quaternion (body-z = −R̂, body-x = Ŝ, body-y = −Ŵ, right-handed)
# DCM (body→LVLH): columns = [[0,1,0], [0,0,-1], [-1,0,0]]
# det = +1; _q_boresight(_NADIR_Q) = [-1, 0, 0]  ✓
_NADIR_Q = np.array([-0.5, 0.5, 0.5, -0.5])


# ---------------------------------------------------------------------------
# AttitudeLaw
# ---------------------------------------------------------------------------

class AttitudeLaw:
    """Spacecraft/sensor pointing law with full quaternion internal storage.

    Prefer the factory classmethods :meth:`fixed`, :meth:`track`, and
    :meth:`nadir` for typical usage.  The constructor is public and may be
    called directly by power users or subclasses.

    Parameters
    ----------
    mode : str
        Pointing mode: ``'fixed'`` or ``'track'``.
    q : ndarray of shape (4,), optional
        Unit quaternion ``[w, x, y, z]`` defining the body orientation in the
        reference frame.  Required for ``mode='fixed'``.
    frame : str, optional
        Reference frame for the quaternion: ``'lvlh'``, ``'eci'``, or
        ``'ecef'``.  Required for ``mode='fixed'``.
    target : Spacecraft, optional
        Target spacecraft to track.  Required for ``mode='track'``.
    roll : float, optional
        Roll angle (rad) about the boresight axis.  Default 0.

    Boresight convention
    --------------------
    The sensor/spacecraft boresight is always **body-z**.  The pointing
    methods return this axis expressed in the requested frame.

    Modes
    -----
    ``'fixed'``
        A constant body orientation in a given reference frame, stored as a
        unit quaternion ``[w, x, y, z]``.
    ``'track'``
        Points from the host spacecraft toward a target
        :class:`~missiontools.Spacecraft`.  The 2-DOF pointing direction is
        fully defined; the roll DOF is reserved for a future release.

    Examples
    --------
    Nadir pointing::

        from missiontools import AttitudeLaw
        law = AttitudeLaw.nadir()
        p_eci = law.pointing_eci(r, v, t)

    Track a target spacecraft::

        from missiontools import Spacecraft, AttitudeLaw
        target = Spacecraft(...)
        law    = AttitudeLaw.track(target)

    Fixed attitude in LVLH (e.g., body-z pointing along-track)::

        law = AttitudeLaw.fixed([0, 1, 0], 'lvlh')
    """

    def __init__(self, mode: str, *,
                 q: npt.NDArray | None = None,
                 frame: str | None = None,
                 target=None,
                 roll: float = 0.0,
                 callback=None):
        self._mode     = mode      # 'fixed' | 'track' | 'custom'
        self._q        = q         # (4,) unit ndarray [w,x,y,z], or None
        self._frame    = frame     # 'lvlh' | 'eci' | 'ecef', or None
        self._target   = target    # Spacecraft, or None
        self._roll     = roll      # roll about boresight for 'track' mode
        self._callback = callback  # callable(t, r_eci, v_eci) -> (N,4), or None
        # Pre-compute boresight direction in the reference frame for 'fixed'
        self._pointing_in_ref = _q_boresight(q) if q is not None else None
        # Yaw steering state
        self._solar_config  = None   # AbstractSolarConfig, or None
        self._yaw_opt_dir   = None   # (3,) body-frame direction to face sun

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def fixed(cls, vector, frame: str,
              roll: float = 0.0) -> AttitudeLaw:
        """Fixed attitude law: constant body orientation in a given frame.

        Parameters
        ----------
        vector : array_like, shape (3,)
            Boresight direction (body-z) expressed in ``frame``.  Need not
            be a unit vector — it is normalised on input.
        frame : {'lvlh', 'eci', 'ecef'}
            Reference frame in which ``vector`` is expressed.
        roll : float, optional
            Roll angle (rad) about the boresight axis.  Default 0 gives the
            minimum rotation from identity that aligns body-z with ``vector``.

        Raises
        ------
        ValueError
            If ``frame`` is not recognised or ``vector`` is the zero vector.
        """
        vec = np.asarray(vector, dtype=np.float64)
        if vec.shape != (3,):
            raise ValueError(
                f"vector must have shape (3,), got {vec.shape}"
            )
        if np.linalg.norm(vec) == 0.0:
            raise ValueError("vector must not be the zero vector")
        if frame not in _VALID_FRAMES:
            raise ValueError(
                f"frame must be one of {sorted(_VALID_FRAMES)}, got {frame!r}"
            )
        q = _q_from_vec(vec, roll)
        return cls('fixed', q=q, frame=frame)

    @classmethod
    def track(cls, target, roll: float = 0.0) -> AttitudeLaw:
        """Target-tracking attitude law: boresight always points toward target.

        Parameters
        ----------
        target : Spacecraft
            The spacecraft to track.
        roll : float, optional
            Roll angle (rad) about the boresight axis.  Default 0 uses the
            minimum-rotation convention from :func:`_q_from_vec` to pin the
            remaining degree of freedom.  Note that ``roll=0`` does **not**
            correspond to a physically meaningful reference orientation such
            as orbit-normal or sun-pointing; it is purely a deterministic
            convention that changes as the target moves.

        Raises
        ------
        TypeError
            If ``target`` is not a :class:`~missiontools.Spacecraft`.
        """
        from ..spacecraft import Spacecraft  # local import avoids circular dep
        if not isinstance(target, Spacecraft):
            raise TypeError(
                f"target must be a Spacecraft instance, "
                f"got {type(target).__name__!r}"
            )
        return cls('track', target=target, roll=roll)

    @classmethod
    def nadir(cls, roll: float = 0.0) -> AttitudeLaw:
        """Earth-nadir pointing law (body-z = −R̂ in LVLH).

        Full 3-DOF convention at ``roll=0``: body-z = nadir (−R̂),
        body-x = along-track (Ŝ), body-y = −orbit-normal (−Ŵ).  This is a
        right-handed body frame.

        Parameters
        ----------
        roll : float, optional
            Roll angle (rad) about the boresight (nadir) axis.  Default 0
            gives body-x = along-track.
        """
        if roll == 0.0:
            q = _NADIR_Q.copy()
        else:
            q_roll = np.array([np.cos(roll / 2), 0., 0., np.sin(roll / 2)])
            q = _q_compose(_NADIR_Q, q_roll)
        return cls('fixed', q=q, frame='lvlh')

    @classmethod
    def custom(cls, callback) -> AttitudeLaw:
        """Custom attitude law defined by a user-supplied callback.

        The callback receives the spacecraft state at each timestep and returns
        full 3-DOF body attitude quaternions in ECI, giving complete control
        over pointing.

        Parameters
        ----------
        callback : callable
            A function with signature::

                callback(t, r_eci, v_eci) -> quaternions

            where:

            * ``t``     — ``(N,)`` array of ``datetime64[us]`` time instants
            * ``r_eci`` — ``(N, 3)`` ECI position array (m)
            * ``v_eci`` — ``(N, 3)`` ECI velocity array (m s⁻¹)
            * returns   — ``(N, 4)`` array of unit quaternions ``[w, x, y, z]``
              defining body attitude in ECI

        Raises
        ------
        TypeError
            If ``callback`` is not callable.

        Examples
        --------
        ::

            def my_attitude(t, r_eci, v_eci):
                N = len(t)
                q = np.zeros((N, 4))
                q[:, 0] = 1.0  # identity: body frame aligned with ECI
                return q

            law = AttitudeLaw.custom(my_attitude)
        """
        if not callable(callback):
            raise TypeError(
                f"callback must be callable, got {type(callback).__name__!r}"
            )
        return cls('custom', callback=callback)

    # ------------------------------------------------------------------
    # Yaw steering
    # ------------------------------------------------------------------

    def yaw_steering(self, solar_config) -> None:
        """Enable or disable solar yaw steering.

        When active, the roll DOF is controlled dynamically at each timestep
        to maximise solar power generation.  The boresight direction is
        unchanged; only the rotation about the boresight is affected.

        Parameters
        ----------
        solar_config : AbstractSolarConfig or None
            Solar config whose :meth:`optimal_angle` defines the preferred
            sun-facing direction in the body frame.  Pass ``None`` to
            deactivate yaw steering and revert to the static roll angle.

        Raises
        ------
        TypeError
            If ``solar_config`` is not an
            :class:`~missiontools.power.AbstractSolarConfig` or ``None``.
        """
        if solar_config is None:
            self._solar_config = None
            self._yaw_opt_dir  = None
            return

        if self._mode == 'custom':
            raise NotImplementedError(
                "yaw_steering() is not supported for 'custom' mode; "
                "incorporate solar optimisation directly in your callback."
            )

        from ..power.solar_config import AbstractSolarConfig
        if not isinstance(solar_config, AbstractSolarConfig):
            raise TypeError(
                f"solar_config must be an AbstractSolarConfig instance or "
                f"None, got {type(solar_config).__name__!r}"
            )

        # Pre-compute the body-frame direction that should face the sun.
        # optimal_angle returns theta measured from basis u=[0,1,0], v=[-1,0,0]
        # (when rotation_axis=[0,0,1]).  The sun-facing direction is -d(theta):
        #   -d = -cos(theta)*u - sin(theta)*v = [sin(theta), -cos(theta), 0]
        theta = solar_config.optimal_angle(np.array([0., 0., 1.]))
        self._yaw_opt_dir  = np.array([
            np.sin(theta), -np.cos(theta), 0.0,
        ])
        self._solar_config = solar_config

    def _compute_yaw_rolls(self, b_eci, r_2d, v_2d, t_arr):
        """Compute per-timestep roll angles for yaw steering.

        Parameters
        ----------
        b_eci : ndarray, shape (N, 3)
            Boresight unit vectors in ECI (independent of roll).
        r_2d, v_2d : ndarray, shape (N, 3)
            Spacecraft ECI position and velocity.
        t_arr : ndarray, shape (N,)
            Timestamps (datetime64[us]).

        Returns
        -------
        ndarray, shape (N,)
            Roll angles (rad) that align the optimal body-frame direction
            with the sun.
        """
        N = len(t_arr)
        s_eci = np.atleast_2d(sun_vec_eci(t_arr))       # (N, 3)

        # At roll=0, rotate the optimal body-frame direction to ref frame
        if self._mode == 'fixed':
            q0 = _q_from_vec(self._pointing_in_ref)      # base quat, roll=0
            d0_ref = _q_rotate(q0, self._yaw_opt_dir)    # (3,) in ref frame
            d0_ref_tiled = np.tile(d0_ref, (N, 1))       # (N, 3)
            if   self._frame == 'eci':  d0_eci = d0_ref_tiled
            elif self._frame == 'lvlh': d0_eci = lvlh_to_eci(d0_ref_tiled, r_2d, v_2d)
            else:                       d0_eci = ecef_to_eci(d0_ref_tiled, t_arr)
        else:  # 'track'
            # b_eci are per-timestep boresight directions; build roll=0 quats
            qs0   = _q_from_vec_batch(b_eci)             # (N, 4) roll=0
            d0_eci = _q_rotate_batch(qs0, self._yaw_opt_dir)  # (N, 3) in ECI

        # Project d0_eci and sun into the plane perpendicular to boresight
        d0_dot_b = np.einsum('ij,ij->i', d0_eci, b_eci)     # (N,)
        d0_perp  = d0_eci - d0_dot_b[:, None] * b_eci       # (N, 3)

        s_dot_b  = np.einsum('ij,ij->i', s_eci, b_eci)      # (N,)
        s_perp   = s_eci - s_dot_b[:, None] * b_eci          # (N, 3)

        # Signed angle from d0_perp to s_perp about b_eci
        cross  = np.cross(d0_perp, s_perp)                   # (N, 3)
        sin_a  = np.einsum('ij,ij->i', cross, b_eci)         # (N,)
        cos_a  = np.einsum('ij,ij->i', d0_perp, s_perp)      # (N,)
        return np.arctan2(sin_a, cos_a)                       # (N,)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._mode == 'fixed':
            return (
                f"AttitudeLaw(mode='fixed', frame={self._frame!r}, "
                f"boresight={self._pointing_in_ref})"
            )
        if self._mode == 'track':
            return f"AttitudeLaw(mode='track', target={self._target!r})"
        if self._mode == 'custom':
            cb_name = getattr(self._callback, '__name__', repr(self._callback))
            return f"AttitudeLaw(mode='custom', callback={cb_name!r})"
        return f"AttitudeLaw(mode={self._mode!r})"

    # ------------------------------------------------------------------
    # Pointing methods
    # ------------------------------------------------------------------

    def pointing_eci(self,
                     r_eci: npt.ArrayLike,
                     v_eci: npt.ArrayLike,
                     t: npt.ArrayLike,
                     ) -> npt.NDArray[np.floating]:
        """Boresight unit vector(s) in the ECI frame.

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
            Unit pointing vector(s) in ECI, shape ``(N, 3)`` for array
            inputs or ``(3,)`` for scalar inputs.
        """
        r     = np.asarray(r_eci, dtype=np.float64)
        scalar = r.ndim == 1
        r_2d  = np.atleast_2d(r)
        v_2d  = np.atleast_2d(np.asarray(v_eci, dtype=np.float64))
        t_arr = np.atleast_1d(np.asarray(t, dtype='datetime64[us]'))
        N     = len(r_2d)

        if self._mode == 'fixed':
            tile = np.tile(self._pointing_in_ref, (N, 1))  # (N, 3)
            if   self._frame == 'eci':  vecs = tile
            elif self._frame == 'lvlh': vecs = lvlh_to_eci(tile, r_2d, v_2d)
            else:                       vecs = ecef_to_eci(tile, t_arr)
        elif self._mode == 'track':
            r_tgt, _ = propagate_analytical(
                t_arr, **self._target.keplerian_params,
                propagator_type=self._target.propagator_type,
            )
            vecs = r_tgt - r_2d
        else:  # 'custom'
            qs   = np.asarray(self._callback(t_arr, r_2d, v_2d), dtype=np.float64)
            vecs = _q_rotate_batch(qs, np.array([0., 0., 1.]))

        result = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return result[0] if scalar else result

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
            Unit pointing vector(s) in LVLH, shape ``(N, 3)`` for array
            inputs or ``(3,)`` for scalar inputs.
        """
        r     = np.asarray(r_eci, dtype=np.float64)
        scalar = r.ndim == 1
        r_2d  = np.atleast_2d(r)
        v_2d  = np.atleast_2d(np.asarray(v_eci, dtype=np.float64))
        eci   = np.atleast_2d(self.pointing_eci(r_eci, v_eci, t))  # (N, 3)
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
            Unit pointing vector(s) in ECEF, shape ``(N, 3)`` for array
            inputs or ``(3,)`` for scalar inputs.
        """
        r     = np.asarray(r_eci, dtype=np.float64)
        scalar = r.ndim == 1
        t_arr = np.atleast_1d(np.asarray(t, dtype='datetime64[us]'))
        eci   = np.atleast_2d(self.pointing_eci(r_eci, v_eci, t))  # (N, 3)
        result = eci_to_ecef(eci, t_arr)
        return result[0] if scalar else result

    def rotate_from_body(self,
                         v_body: npt.ArrayLike,
                         r_eci:  npt.ArrayLike,
                         v_eci:  npt.ArrayLike,
                         t:      npt.ArrayLike,
                         ) -> npt.NDArray[np.floating]:
        """Express a spacecraft body-frame vector in the ECI frame.

        Supported for both ``'fixed'`` and ``'track'`` modes.  For ``'track'``
        mode the :attr:`roll` angle stored on the law (default 0) pins the
        remaining degree of freedom using the minimum-rotation convention.

        Parameters
        ----------
        v_body : array_like, shape (3,)
            Unit direction in the spacecraft body frame (need not be
            pre-normalised).
        r_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI position(s) (m).
        v_eci : array_like, shape ``(N, 3)`` or ``(3,)``
            Host spacecraft ECI velocity(s) (m s⁻¹).
        t : array_like of datetime64, shape ``(N,)`` or scalar
            Observation epoch(s).

        Returns
        -------
        npt.NDArray[np.floating]
            Unit vector(s) in ECI, shape ``(N, 3)`` or ``(3,)``.
        """
        v   = np.asarray(v_body, dtype=np.float64)
        v   = v / np.linalg.norm(v)
        r   = np.asarray(r_eci, dtype=np.float64)
        scalar = r.ndim == 1
        r_2d  = np.atleast_2d(r)
        v_2d  = np.atleast_2d(np.asarray(v_eci, dtype=np.float64))
        t_arr = np.atleast_1d(np.asarray(t, dtype='datetime64[us]'))
        N     = len(r_2d)

        if self._mode == 'fixed':
            if self._solar_config is not None:
                # Yaw steering: per-timestep roll
                boresight = self._pointing_in_ref              # (3,) in ref frame
                b_ref = np.tile(boresight, (N, 1))             # (N, 3) in ref
                if   self._frame == 'eci':  b_eci = b_ref
                elif self._frame == 'lvlh': b_eci = lvlh_to_eci(b_ref, r_2d, v_2d)
                else:                       b_eci = ecef_to_eci(b_ref, t_arr)
                b_eci = b_eci / np.linalg.norm(b_eci, axis=1, keepdims=True)
                yaw_rolls = self._compute_yaw_rolls(b_eci, r_2d, v_2d, t_arr)
                qs = _q_from_vec_batch(
                    np.tile(boresight, (N, 1)), roll=yaw_rolls,
                )                                              # (N, 4)
                vecs_ref = _q_rotate_batch(qs, v)              # (N, 3) in ref
                if   self._frame == 'eci':  vecs = vecs_ref
                elif self._frame == 'lvlh': vecs = lvlh_to_eci(vecs_ref, r_2d, v_2d)
                else:                       vecs = ecef_to_eci(vecs_ref, t_arr)
            else:
                v_ref = _q_rotate(self._q, v)                  # (3,) in ref frame
                tiled = np.tile(v_ref, (N, 1))                 # (N, 3)
                if   self._frame == 'eci':  vecs = tiled
                elif self._frame == 'lvlh': vecs = lvlh_to_eci(tiled, r_2d, v_2d)
                else:                       vecs = ecef_to_eci(tiled, t_arr)
        elif self._mode == 'track':
            r_tgt, _ = propagate_analytical(
                t_arr, **self._target.keplerian_params,
                propagator_type=self._target.propagator_type,
            )
            d    = r_tgt - r_2d
            d    = d / np.linalg.norm(d, axis=1, keepdims=True)  # (N, 3) unit
            if self._solar_config is not None:
                b_eci = d  # boresight in ECI = direction to target
                yaw_rolls = self._compute_yaw_rolls(b_eci, r_2d, v_2d, t_arr)
                qs = _q_from_vec_batch(d, roll=yaw_rolls)
            else:
                qs = _q_from_vec_batch(d, roll=self._roll)
            vecs = _q_rotate_batch(qs, v)                         # (N, 3)
        else:  # 'custom'
            qs   = np.asarray(self._callback(t_arr, r_2d, v_2d), dtype=np.float64)
            vecs = _q_rotate_batch(qs, v)                         # (N, 3)

        result = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return result[0] if scalar else result
