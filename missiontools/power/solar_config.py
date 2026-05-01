"""Solar power configuration classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from ..orbit.frames import sun_vec_eci
from ..orbit.propagation import propagate_analytical
from ..orbit.shadow import in_sunlight


class AbstractSolarConfig(ABC):
    """Base class for solar power configurations.

    A solar config models the power generation of a spacecraft's solar arrays.
    It must be attached to a :class:`~missiontools.Spacecraft` via
    :meth:`~missiontools.Spacecraft.add_solar_config` before calling methods
    that require orbital state.

    Parameters
    ----------
    efficiency : float
        Solar-to-DC conversion efficiency, in (0, 1].
    """

    def __init__(self, efficiency: float) -> None:
        if not 0 < efficiency <= 1:
            raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
        self._efficiency = float(efficiency)
        self._spacecraft = None

    @property
    def efficiency(self) -> float:
        """Solar-to-DC conversion efficiency."""
        return self._efficiency

    @property
    def spacecraft(self):
        """Spacecraft this config is attached to, or ``None``."""
        return self._spacecraft

    def _require_spacecraft(self) -> None:
        if self._spacecraft is None:
            raise RuntimeError(
                "Solar config must be attached to a Spacecraft via "
                "add_solar_config() before calling this method."
            )

    @abstractmethod
    def generation(
        self,
        t_start: np.datetime64,
        t_end: np.datetime64,
        step: np.timedelta64,
        irradiance: float = 1366.0,
    ) -> dict:
        """Compute instantaneous solar power generation.

        Parameters
        ----------
        t_start : datetime64
            Start of the time window.
        t_end : datetime64
            End of the time window (inclusive).
        step : timedelta64
            Time step between samples.
        irradiance : float, optional
            Solar irradiance (W m⁻²).  Defaults to AM0 constant, 1366 W m⁻².

        Returns
        -------
        dict
            ``t`` : ``(N,)`` ``datetime64[us]`` — sample timestamps.

            ``power`` : ``(N,)`` float — instantaneous power (W).
        """

    @abstractmethod
    def optimal_angle(self, rotation_axis: npt.ArrayLike) -> float:
        """Return the body-frame rotation angle that maximises projected area.

        Parameters
        ----------
        rotation_axis : array_like, shape (3,)
            Rotation axis in the spacecraft body frame.

        Returns
        -------
        float
            Optimal angle (rad) measured from a deterministic reference
            direction in the plane perpendicular to *rotation_axis*.
        """

    @abstractmethod
    def oap(self, start_time: np.datetime64 | None = None) -> float:
        """Orbit-average power over one orbital period.

        Parameters
        ----------
        start_time : datetime64, optional
            Start of the averaging window.  Defaults to the spacecraft epoch.

        Returns
        -------
        float
            Mean power (W) over one orbit.
        """


class NormalVectorSolarConfig(AbstractSolarConfig):
    """Solar config defined by panel normal vectors and areas.

    Power for each panel is ``irradiance * area * efficiency *
    max(n̂_eci · ŝ_eci, 0)`` where *n̂_eci* is the panel outward normal
    rotated into ECI via the spacecraft attitude law and *ŝ_eci* is the
    unit vector toward the Sun.  No self-shadowing is modelled.

    Parameters
    ----------
    normal_vecs : array_like, shape (M, 3)
        Outward-facing normal vectors of each panel in the spacecraft body
        frame.  They are normalised internally.
    areas : array_like, shape (M,)
        Panel areas (m²).
    efficiency : float
        Solar-to-DC conversion efficiency, in (0, 1].
    """

    def __init__(
        self,
        normal_vecs: npt.ArrayLike,
        areas: npt.ArrayLike,
        efficiency: float,
    ) -> None:
        super().__init__(efficiency)
        normals = np.asarray(normal_vecs, dtype=np.float64)
        areas_arr = np.asarray(areas, dtype=np.float64)

        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError(f"normal_vecs must have shape (M, 3), got {normals.shape}")
        if areas_arr.ndim != 1 or len(areas_arr) != len(normals):
            raise ValueError(
                f"areas must have shape ({len(normals)},), got {areas_arr.shape}"
            )
        if np.any(areas_arr <= 0):
            raise ValueError("All panel areas must be positive.")

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Normal vectors must be non-zero.")

        self._normals = normals / norms  # (M, 3) unit normals
        self._areas = areas_arr  # (M,)

    @property
    def normals(self) -> npt.NDArray[np.floating]:
        """Panel unit normals in the body frame, shape ``(M, 3)``."""
        return self._normals.copy()

    @property
    def areas(self) -> npt.NDArray[np.floating]:
        """Panel areas (m²), shape ``(M,)``."""
        return self._areas.copy()

    def generation(
        self,
        t_start: np.datetime64,
        t_end: np.datetime64,
        step: np.timedelta64,
        irradiance: float = 1366.0,
    ) -> dict:
        """Compute instantaneous solar power generation.

        Instantaneous power from panel *k* is

        .. math::

            P_k = I \\cdot A_k \\cdot \\eta \\cdot \\max(\\hat{n}_{k,\\mathrm{ECI}}
                  \\cdot \\hat{s}_{\\mathrm{ECI}},\\; 0)

        where :math:`I` is the solar irradiance, :math:`A_k` is the panel
        area, :math:`\\eta` is the conversion efficiency,
        :math:`\\hat{n}_{k,\\mathrm{ECI}}` is the panel outward normal rotated
        into ECI via the spacecraft attitude law, and
        :math:`\\hat{s}_{\\mathrm{ECI}}` is the unit vector toward the Sun.
        Power is set to zero in eclipse.

        Parameters
        ----------
        t_start : np.datetime64
            Start of the time window.
        t_end : np.datetime64
            End of the time window (inclusive).
        step : np.timedelta64
            Time step between samples.
        irradiance : float, optional
            Solar irradiance (W m⁻²).  Defaults to the AM0 solar constant,
            1366 W m⁻².

        Returns
        -------
        dict
            ``'t'`` : ``(N,)`` ``datetime64[us]`` — sample timestamps.

            ``'power'`` : ``(N,)`` float — instantaneous total power (W).

        Raises
        ------
        RuntimeError
            If the config has not been attached to a spacecraft via
            :meth:`~missiontools.Spacecraft.add_solar_config`.
        """
        self._require_spacecraft()
        sc = self._spacecraft

        state = sc.propagate(t_start, t_end, step)
        t = state["t"]
        r = state["r"]  # (N, 3)
        v = state["v"]  # (N, 3)

        if len(t) == 0:
            return {
                "t": np.array([], dtype="datetime64[us]"),
                "power": np.empty(0, dtype=np.float64),
            }

        sun = sun_vec_eci(t)  # (N, 3)
        lit = in_sunlight(r, t, body_radius=sc.central_body_radius)  # (N,)

        N = len(t)
        power = np.zeros(N, dtype=np.float64)

        for k, (normal, area) in enumerate(zip(self._normals, self._areas)):
            # Transform body-frame normal to ECI
            n_eci = sc.attitude_law.rotate_from_body(
                normal,
                r,
                v,
                t,
            )  # (N, 3)
            n_eci_2d = np.atleast_2d(n_eci)  # ensure (N, 3)
            cos_angle = np.einsum("ij,ij->i", n_eci_2d, np.atleast_2d(sun))
            power += area * np.maximum(cos_angle, 0.0)

        power *= irradiance * self._efficiency
        power[~lit] = 0.0

        return {"t": t, "power": power}

    def optimal_angle(self, rotation_axis: npt.ArrayLike) -> float:
        """Return the rotation angle that maximises total projected panel area.

        Searches 360 equally-spaced candidate orientations about
        ``rotation_axis`` and returns the angle :math:`\\theta` (measured from
        a deterministic reference direction in the perpendicular plane) at
        which the sum of area-weighted :math:`\\max(\\hat{n}_k \\cdot \\hat{s}, 0)`
        over all panels is greatest.

        Parameters
        ----------
        rotation_axis : array_like, shape (3,)
            Rotation axis in the spacecraft body frame (need not be a unit
            vector).

        Returns
        -------
        float
            Optimal angle (rad) in :math:`[0, 2\\pi)` measured from the
            reference direction ``u = \\mathrm{axis} \\times e_{\\min}``
            where :math:`e_{\\min}` is the cardinal axis least aligned with
            ``rotation_axis``.
        """
        axis = np.asarray(rotation_axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)

        # Build orthonormal basis in the plane perpendicular to axis.
        # Choose the cardinal axis least aligned with the rotation axis.
        cardinals = np.eye(3)
        dots = np.abs(cardinals @ axis)
        least = cardinals[np.argmin(dots)]
        u = np.cross(axis, least)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)

        n_samples = 360
        thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

        # Candidate sun-direction vectors: (n_samples, 3)
        # Sun comes FROM -d, so contribution = max(-d · normal, 0)
        d = np.outer(np.cos(thetas), u) + np.outer(np.sin(thetas), v)  # (S, 3)
        # dot[s, m] = (-d[s]) · normal[m] = -d[s] @ normals.T
        dots = -d @ self._normals.T  # (S, M)
        # Weighted sum: (S, M) clipped × areas (M,) → (S,)
        total = np.maximum(dots, 0.0) @ self._areas  # (S,)

        return float(thetas[np.argmax(total)])

    def oap(self, start_time: np.datetime64 | None = None) -> float:
        """Orbit-average power over one orbital period.

        Propagates the orbit for exactly one Keplerian period starting from
        ``start_time`` and returns the time-mean of the instantaneous power
        computed by :meth:`generation`.

        Parameters
        ----------
        start_time : np.datetime64, optional
            Start of the averaging window.  Defaults to the spacecraft epoch.

        Returns
        -------
        float
            Mean power (W) over one orbit.

        Raises
        ------
        RuntimeError
            If the config has not been attached to a spacecraft via
            :meth:`~missiontools.Spacecraft.add_solar_config`.
        """
        self._require_spacecraft()
        sc = self._spacecraft

        mu = sc.central_body_mu
        period_s = 2 * np.pi * np.sqrt(sc.a**3 / mu)
        period = np.timedelta64(int(period_s * 1e6), "us")

        t0_src = sc.epoch if start_time is None else start_time
        t0 = np.asarray(t0_src, dtype="datetime64[us]")

        step_s = min(30.0, period_s / 360.0)
        step = np.timedelta64(int(step_s * 1e6), "us")

        result = self.generation(t0, t0 + period, step)
        return float(np.mean(result["power"]))
