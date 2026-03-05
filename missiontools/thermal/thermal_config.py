"""Thermal surface configuration classes.

A thermal config models the radiative interaction of a spacecraft's external
surfaces with the space environment (solar absorption, infrared emission).
Attach it to a :class:`~missiontools.Spacecraft` via
:meth:`~missiontools.Spacecraft.add_thermal_config`, then couple its faces
to a :class:`ThermalCircuit` via :meth:`attach`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from ..orbit.frames import sun_vec_eci
from ..orbit.shadow import in_sunlight

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)


class AbstractThermalConfig(ABC):
    """Base class for thermal surface configurations.

    Stores per-face areas, emissivities, and absorptivities.  Subclasses
    implement :meth:`_compute_absorbed_solar` to define how incident solar
    flux is projected onto each face.  The concrete :meth:`attach` method
    pre-computes solar absorption over a time span and creates load
    functions (combining absorption with T⁴ emission) that are added to a
    :class:`ThermalCircuit`.

    Parameters
    ----------
    areas : array_like, shape (M,)
        Face areas (m²).  Must be positive.
    emissivities : array_like, shape (M,)
        IR emissivity of each face, in [0, 1].
    absorptivities : array_like, shape (M,)
        Solar absorptivity of each face, in [0, 1].
    irradiance : float
        Solar irradiance (W m⁻²).  Defaults to AM0 constant, 1366 W m⁻².
    """

    def __init__(
        self,
        areas: npt.ArrayLike,
        emissivities: npt.ArrayLike,
        absorptivities: npt.ArrayLike,
        irradiance: float = 1366.0,
    ) -> None:
        areas_arr = np.asarray(areas, dtype=np.float64)
        emissivities_arr = np.asarray(emissivities, dtype=np.float64)
        absorptivities_arr = np.asarray(absorptivities, dtype=np.float64)

        if areas_arr.ndim != 1:
            raise ValueError(
                f"areas must be 1-D, got shape {areas_arr.shape}"
            )
        m = len(areas_arr)
        if emissivities_arr.shape != (m,):
            raise ValueError(
                f"emissivities must have shape ({m},), "
                f"got {emissivities_arr.shape}"
            )
        if absorptivities_arr.shape != (m,):
            raise ValueError(
                f"absorptivities must have shape ({m},), "
                f"got {absorptivities_arr.shape}"
            )
        if np.any(areas_arr <= 0):
            raise ValueError("All face areas must be positive.")
        if np.any((emissivities_arr < 0) | (emissivities_arr > 1)):
            raise ValueError("Emissivities must be in [0, 1].")
        if np.any((absorptivities_arr < 0) | (absorptivities_arr > 1)):
            raise ValueError("Absorptivities must be in [0, 1].")

        irradiance = float(irradiance)
        if irradiance <= 0:
            raise ValueError(
                f"irradiance must be positive, got {irradiance}."
            )

        self._areas = areas_arr
        self._emissivities = emissivities_arr
        self._absorptivities = absorptivities_arr
        self._irradiance = irradiance
        self._spacecraft = None

    # --- properties ---

    @property
    def areas(self) -> npt.NDArray[np.floating]:
        """Face areas (m²), shape ``(M,)``."""
        return self._areas.copy()

    @property
    def emissivities(self) -> npt.NDArray[np.floating]:
        """IR emissivity per face, shape ``(M,)``."""
        return self._emissivities.copy()

    @property
    def absorptivities(self) -> npt.NDArray[np.floating]:
        """Solar absorptivity per face, shape ``(M,)``."""
        return self._absorptivities.copy()

    @property
    def irradiance(self) -> float:
        """Solar irradiance (W m⁻²)."""
        return self._irradiance

    @property
    def num_faces(self) -> int:
        """Number of faces."""
        return len(self._areas)

    @property
    def spacecraft(self):
        """Spacecraft this config is attached to, or ``None``."""
        return self._spacecraft

    def _require_spacecraft(self) -> None:
        if self._spacecraft is None:
            raise RuntimeError(
                "Thermal config must be attached to a Spacecraft via "
                "add_thermal_config() before calling this method."
            )

    # --- abstract interface ---

    @abstractmethod
    def _compute_absorbed_solar(
        self,
        r: np.ndarray,
        v: np.ndarray,
        t: np.ndarray,
        sun_eci: np.ndarray,
        lit: np.ndarray,
    ) -> np.ndarray:
        """Compute absorbed solar power for each face at each timestep.

        Parameters
        ----------
        r : ndarray, shape (N, 3)
            ECI position (m).
        v : ndarray, shape (N, 3)
            ECI velocity (m/s).
        t : ndarray, shape (N,)
            Timestamps (datetime64[us]).
        sun_eci : ndarray, shape (N, 3)
            Unit vectors toward the Sun in ECI.
        lit : ndarray, shape (N,)
            Boolean mask: True where spacecraft is in sunlight.

        Returns
        -------
        ndarray, shape (N, M)
            Absorbed solar power (W) per face per timestep.
        """

    # --- circuit coupling ---

    def attach(
        self,
        circuit,
        face_nodes: list[str],
        t_start: np.datetime64,
        t_end: np.datetime64,
        step: np.timedelta64,
        *,
        prefix: str = 'thermal',
    ) -> float:
        """Couple surface faces to a :class:`ThermalCircuit`.

        Pre-computes solar absorption over the time span, then creates
        load functions for each face that combine interpolated absorption
        with Stefan-Boltzmann emission (``-ε σ A T⁴``).  Each load is
        registered on the circuit via
        :meth:`~ThermalCircuit.add_load`.

        Parameters
        ----------
        circuit : ThermalCircuit
            The thermal circuit to attach loads to.
        face_nodes : list of str
            Capacitance node name for each face.  Length must equal
            :attr:`num_faces`.
        t_start : datetime64
            Start of the simulation window.
        t_end : datetime64
            End of the simulation window.
        step : timedelta64
            Time step for orbital propagation (used to pre-compute
            solar absorption).
        prefix : str
            Name prefix for the load elements added to the circuit.
            Defaults to ``'thermal'``.

        Returns
        -------
        float
            Simulation duration in seconds, for passing to
            ``circuit.solve()``.
        """
        self._require_spacecraft()
        sc = self._spacecraft

        if len(face_nodes) != self.num_faces:
            raise ValueError(
                f"face_nodes length ({len(face_nodes)}) must match "
                f"num_faces ({self.num_faces})."
            )

        # Propagate orbit
        state = sc.propagate(t_start, t_end, step)
        t = state['t']
        r = state['r']
        v = state['v']

        if len(t) == 0:
            return 0.0

        # Compute orbital environment
        sun = sun_vec_eci(t)
        lit = in_sunlight(r, t, body_radius=sc.central_body_radius)

        # Pre-compute absorbed solar power: (N, M)
        absorbed = self._compute_absorbed_solar(r, v, t, sun, lit)

        # Convert timestamps to seconds from zero
        t_sec = (t - t[0]) / np.timedelta64(1, 'us') * 1e-6
        duration = float(t_sec[-1])

        # Create load functions and register on circuit
        for m in range(self.num_faces):
            absorbed_m = absorbed[:, m].copy()
            t_sec_m = t_sec.copy()
            eps_m = float(self._emissivities[m])
            area_m = float(self._areas[m])

            def _make_load_fn(t_arr, q_arr, eps, area):
                def load_fn(t, T):
                    q_solar = np.interp(t, t_arr, q_arr)
                    q_emit = eps * STEFAN_BOLTZMANN * area * T ** 4
                    return q_solar - q_emit
                return load_fn

            fn = _make_load_fn(t_sec_m, absorbed_m, eps_m, area_m)
            circuit.add_load(f'{prefix}_face_{m}', face_nodes[m], fn)

        return duration


class NormalVectorThermalConfig(AbstractThermalConfig):
    """Thermal config defined by face normal vectors.

    Each face is characterised by an outward-facing normal vector in the
    spacecraft body frame, an area, an IR emissivity, and a solar
    absorptivity.  Solar absorption on face *m* is

    .. math::

        Q_{\\mathrm{abs},m} = \\alpha_m \\, A_m \\, \\max(\\hat{n}_{m,\\mathrm{ECI}}
        \\cdot \\hat{s}_{\\mathrm{ECI}},\\; 0) \\, S

    where *S* is the solar irradiance.  Power is zero in eclipse.

    Parameters
    ----------
    normal_vecs : array_like, shape (M, 3)
        Outward-facing normal vectors in the spacecraft body frame.
        Normalised internally.
    areas : array_like, shape (M,)
        Face areas (m²).
    emissivities : array_like, shape (M,)
        IR emissivity of each face, in [0, 1].
    absorptivities : array_like, shape (M,)
        Solar absorptivity of each face, in [0, 1].
    irradiance : float
        Solar irradiance (W m⁻²).  Defaults to 1366 W m⁻².
    """

    def __init__(
        self,
        normal_vecs: npt.ArrayLike,
        areas: npt.ArrayLike,
        emissivities: npt.ArrayLike,
        absorptivities: npt.ArrayLike,
        irradiance: float = 1366.0,
    ) -> None:
        super().__init__(areas, emissivities, absorptivities, irradiance)

        normals = np.asarray(normal_vecs, dtype=np.float64)
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError(
                f"normal_vecs must have shape (M, 3), got {normals.shape}"
            )
        if len(normals) != self.num_faces:
            raise ValueError(
                f"normal_vecs length ({len(normals)}) must match "
                f"areas length ({self.num_faces})."
            )

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Normal vectors must be non-zero.")

        self._normals = normals / norms

    @property
    def normals(self) -> npt.NDArray[np.floating]:
        """Face unit normals in the body frame, shape ``(M, 3)``."""
        return self._normals.copy()

    def _compute_absorbed_solar(
        self,
        r: np.ndarray,
        v: np.ndarray,
        t: np.ndarray,
        sun_eci: np.ndarray,
        lit: np.ndarray,
    ) -> np.ndarray:
        sc = self._spacecraft
        n = len(t)
        m = self.num_faces
        absorbed = np.zeros((n, m), dtype=np.float64)

        sun_2d = np.atleast_2d(sun_eci)

        for k in range(m):
            n_eci = sc.attitude_law.rotate_from_body(
                self._normals[k], r, v, t,
            )
            n_eci_2d = np.atleast_2d(n_eci)
            cos_angle = np.einsum('ij,ij->i', n_eci_2d, sun_2d)
            absorbed[:, k] = (
                self._absorptivities[k]
                * self._areas[k]
                * np.maximum(cos_angle, 0.0)
                * self._irradiance
            )
            absorbed[~lit, k] = 0.0

        return absorbed
