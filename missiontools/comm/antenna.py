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
from ..orbit.constants import EARTH_MEAN_RADIUS
from ..sensor.sensor_law import _euler_zyx_to_boresight


class AbstractAntenna(ABC):
    """Base class for antennas attachable to Spacecraft or GroundStation.

    Mounting is specified via keyword arguments.  Exactly one mounting
    group must be provided (spacecraft or ground station), unless the
    subclass is direction-independent (e.g. :class:`IsotropicAntenna`).

    **Spacecraft mounting** (provide exactly one):

    - ``attitude_law`` — independent :class:`~missiontools.AbstractAttitudeLaw`
    - ``body_vector`` — boresight direction in the spacecraft body frame
    - ``body_euler_deg`` — ``(yaw, pitch, roll)`` ZYX intrinsic Euler
      angles defining the boresight in the body frame

    **Ground station mounting**:

    - ``azimuth_deg`` — azimuth from north (deg), clockwise positive
    - ``elevation_deg`` — elevation from horizon (deg)
    - ``rotation_deg`` — boresight rotation (deg), default 0

    Parameters
    ----------
    attitude_law : AbstractAttitudeLaw, optional
    body_vector : array_like, shape (3,), optional
    body_euler_deg : tuple of float, optional
    azimuth_deg : float, optional
    elevation_deg : float, optional
    rotation_deg : float, optional
    """

    _requires_mounting = True

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
        sc_opts = sum(
            x is not None for x in (attitude_law, body_vector, body_euler_deg)
        )
        gs_opts = azimuth_deg is not None

        if sc_opts and gs_opts:
            raise ValueError(
                "Cannot mix spacecraft mounting (attitude_law / body_vector "
                "/ body_euler_deg) with ground station mounting "
                "(azimuth_deg / elevation_deg)."
            )

        if not sc_opts and not gs_opts:
            if self._requires_mounting:
                raise ValueError(
                    "Must specify spacecraft mounting (attitude_law, "
                    "body_vector, or body_euler_deg) or ground station "
                    "mounting (azimuth_deg + elevation_deg)."
                )
            self._mode = None
            self._spacecraft = None
            self._ground_station = None
            return

        if sc_opts > 1:
            raise ValueError(
                "Specify exactly one of attitude_law, body_vector, or body_euler_deg."
            )

        self._spacecraft = None
        self._ground_station = None

        if gs_opts:
            # Ground station mounting
            if elevation_deg is None:
                raise ValueError(
                    "elevation_deg is required for ground station mounting."
                )
            if not -90.0 <= float(elevation_deg) <= 90.0:
                raise ValueError(
                    f"elevation_deg must be in [-90, 90], got {elevation_deg}"
                )
            self._mode = "ground"
            az_rad = np.radians(float(azimuth_deg))
            el_rad = np.radians(float(elevation_deg))
            self._boresight_enu = azel_to_enu(az_rad, el_rad)
            self._rotation_deg = float(rotation_deg)
            self._boresight_ecef = None  # set at attachment time
        elif attitude_law is not None:
            self._mode = "independent"
            self._attitude_law = attitude_law
        else:
            # Body-mounted
            self._mode = "body"
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
        if self._mode == "independent":
            return self._attitude_law.pointing_eci(r_eci, v_eci, t)
        elif self._mode == "body":
            if self._spacecraft is None:
                raise RuntimeError(
                    "Body-mounted antenna must be attached to a Spacecraft "
                    "via add_antenna() before computing boresight."
                )
            return self._spacecraft.attitude_law.rotate_from_body(
                self._body_vector,
                r_eci,
                v_eci,
                t,
            )
        elif self._mode == "ground":
            if self._boresight_ecef is None:
                raise RuntimeError(
                    "Ground-mounted antenna must be attached to a "
                    "GroundStation via add_antenna() before computing "
                    "boresight."
                )
            t_arr = np.atleast_1d(np.asarray(t, dtype="datetime64[us]"))
            n = len(t_arr)
            boresight_tiled = np.tile(self._boresight_ecef, (n, 1))
            result = ecef_to_eci(boresight_tiled, t_arr)
            if np.asarray(t).ndim == 0:
                return result[0]
            return result
        else:
            raise RuntimeError(f"Cannot compute boresight for mode '{self._mode}'.")

    # --- gain computation ---

    def gain(
        self,
        t: npt.ArrayLike,
        v: npt.ArrayLike,
        frame: str = "eci",
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
        t_arr = np.atleast_1d(np.asarray(t, dtype="datetime64[us]"))
        v_arr = np.atleast_2d(np.asarray(v, dtype=np.float64))
        n = len(v_arr)

        # Convert v to ECI
        if frame == "eci":
            v_eci_dir = v_arr
        elif frame == "ecef":
            v_eci_dir = ecef_to_eci(v_arr, t_arr)
        elif frame == "lvlh":
            if r_eci is None or v_eci is None:
                raise ValueError("r_eci and v_eci are required for frame='lvlh'.")
            v_eci_dir = lvlh_to_eci(v_arr, r_eci, v_eci)
        else:
            raise ValueError(f"Unknown frame '{frame}'. Use 'eci', 'ecef', or 'lvlh'.")

        # Normalise direction vectors
        norms = np.linalg.norm(v_eci_dir, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        v_hat = v_eci_dir / norms

        # Get boresight in ECI
        boresight = np.atleast_2d(self.boresight_eci(r_eci, v_eci, t_arr))

        # Off-boresight angle
        cos_theta = np.einsum("ij,ij->i", boresight, v_hat)
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

    _requires_mounting = False

    def __init__(self, gain_dbi: float = 0.0) -> None:
        super().__init__()
        self._gain_dbi = float(gain_dbi)

    def gain(
        self,
        t: npt.ArrayLike,
        v: npt.ArrayLike,
        frame: str = "eci",
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

    Notes
    -----
    Gain values are linearly interpolated using ``numpy.interp``, which
    **clamps** angles outside the tabulated range to the nearest endpoint
    value.  If the table only covers ``[0°, 90°]``, angles beyond 90° will
    return the gain at 90° rather than a lower back-hemisphere value, which
    may overstate gain in that region.  Extend the table to 180° (e.g. with
    a low back-lobe gain) to avoid this clamping.
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
            raise ValueError(f"angles_deg must be 1-D, got shape {angles.shape}")
        if gains.ndim != 1:
            raise ValueError(f"gains_dbi must be 1-D, got shape {gains.shape}")
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
            raise ValueError("angles_deg must be within [0, 180] degrees.")

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

    # --- factory classmethods ---

    @classmethod
    def from_isoflux(
        cls,
        altitude_km: float,
        min_elev_deg: float = 5.0,
        edge_gain: float | None = None,
        central_body_radius: float = EARTH_MEAN_RADIUS,
        **kwargs,
    ) -> "SymmetricAntenna":
        """Isoflux antenna pattern for a fixed nadir-pointing orbit altitude.

        Shapes the beam so that power flux density at the spherical body
        surface is constant across the coverage footprint.  The gain
        increases from boresight (nadir) toward the edge of coverage to
        compensate for the increasing slant range.

        Parameters
        ----------
        altitude_km : float
            Orbital altitude above the body surface (km).
        min_elev_deg : float, optional
            Minimum surface elevation angle defining the coverage edge (deg).
            Default 5.0°.
        edge_gain : float or None, optional
            Desired gain at the edge of coverage (dBi).  If *None* (default),
            the boresight gain is derived from the constraint that the
            total antenna directivity equals unity (0 dBi on average),
            assuming zero radiation beyond the coverage zone.
        central_body_radius : float, optional
            Mean radius of the central body (m).  Defaults to
            ``EARTH_MEAN_RADIUS``.
        **kwargs
            Mounting keyword arguments forwarded to :class:`SymmetricAntenna`.
        """
        h = float(altitude_km) * 1e3
        R = float(central_body_radius)
        d = R + h

        el_min_rad = np.radians(float(min_elev_deg))
        theta_max = np.arcsin(np.clip((R / d) * np.cos(el_min_rad), -1.0, 1.0))

        # Slant range at each off-nadir angle
        n_main = 200
        thetas = np.linspace(0.0, theta_max, n_main)
        ranges = d * np.cos(thetas) - np.sqrt(
            np.maximum(0.0, R**2 - d**2 * np.sin(thetas) ** 2)
        )

        # Relative gain shape (dB), normalised to boresight = 0
        gain_shape_db = 20.0 * np.log10(ranges / h)

        if edge_gain is not None:
            # Back-compute boresight gain from specified edge gain
            g0 = float(edge_gain) - gain_shape_db[-1]
        else:
            # Unity directivity: D₀ = 2 / ∫₀^θ_max [(r/h)² · sin(θ)] dθ
            integrand = (ranges / h) ** 2 * np.sin(thetas)
            integral = np.trapz(integrand, thetas)
            g0 = 10.0 * np.log10(2.0 / integral)

        gains_main = g0 + gain_shape_db
        computed_edge = float(gains_main[-1])

        # Immediate rolloff just beyond θ_max so np.interp doesn't clamp to
        # the edge gain for out-of-coverage angles.  Use a tiny 0.01° step to
        # keep the linear transition zone negligibly small.
        rolloff_angle = min(theta_max + np.radians(0.01), np.radians(90.0))
        angles_out = np.concatenate([thetas, [rolloff_angle, np.radians(90.0)]])
        gains_out = np.concatenate([gains_main, [-60.0, -60.0]])

        # Remove duplicate angles (e.g. when theta_max is already near 90°)
        _, unique_idx = np.unique(angles_out, return_index=True)
        angles_out = angles_out[unique_idx]
        gains_out = gains_out[unique_idx]

        return cls(np.degrees(angles_out), gains_out, **kwargs)

    @classmethod
    def from_gaussian(
        cls,
        gain_dbi: float,
        **kwargs,
    ) -> "SymmetricAntenna":
        """Ideal Gaussian beam pattern with automatically scaled beamwidth.

        The half-power beamwidth is derived from the normalisation condition
        that total directivity equals *gain_dbi* (i.e. the Gaussian integral
        over the full sphere gives the correct peak value).

        Parameters
        ----------
        gain_dbi : float
            Peak gain at boresight (dBi).
        **kwargs
            Mounting keyword arguments forwarded to :class:`SymmetricAntenna`.
        """
        from scipy.optimize import brentq

        gain_dbi = float(gain_dbi)
        if gain_dbi <= 0.0:
            raise ValueError(
                f"gain_dbi must be positive for a Gaussian beam (got {gain_dbi}). "
                "A Gaussian pattern always has peak directivity > 0 dBi; "
                "use IsotropicAntenna for gain_dbi ≤ 0."
            )

        D = 10.0 ** (gain_dbi / 10.0)  # linear directivity

        def _residual(sigma: float) -> float:
            # D = 2 / ∫₀^π exp(−θ²/(2σ²)) · sin(θ) dθ
            th = np.linspace(0.0, np.pi, 4000)
            integ = np.trapz(np.exp(-(th**2) / (2.0 * sigma**2)) * np.sin(th), th)
            return 2.0 / integ - D

        sigma = brentq(_residual, 1e-4, 50.0)

        # Angle at which gain drops to −60 dBi
        theta_cutoff = sigma * np.sqrt(
            2.0 * (float(gain_dbi) + 60.0) * np.log(10.0) / 10.0
        )
        theta_cutoff = min(theta_cutoff, np.pi / 2.0)

        thetas = np.linspace(0.0, theta_cutoff, 500)
        gains = float(gain_dbi) + 10.0 * np.log10(
            np.maximum(np.exp(-(thetas**2) / (2.0 * sigma**2)), 1e-20)
        )
        # Append a far-field point at 180° to cover the full back hemisphere
        angles_out = np.append(np.degrees(thetas), 180.0)
        gains_out = np.append(gains, -60.0)

        return cls(angles_out, gains_out, **kwargs)

    @classmethod
    def from_parabolic(
        cls,
        diameter: float,
        frequency: float,
        eff: float = 0.6,
        envelope: bool = False,
        **kwargs,
    ) -> "SymmetricAntenna":
        """Uniformly illuminated parabolic reflector antenna pattern.

        Parameters
        ----------
        diameter : float
            Reflector diameter (m).
        frequency : float
            Centre frequency (Hz).
        eff : float, optional
            Antenna efficiency (dimensionless, 0 < eff ≤ 1).  Accounts for
            spillover, blockage, surface errors, etc.  Default 0.6.
        envelope : bool, optional
            If *False* (default), the full pattern including sidelobes is
            returned.  If *True*, the sidelobe envelope is used: the main
            lobe is exact and sidelobes beyond the first null are replaced
            by the asymptotic envelope
            ``f_env(u) = 8 / (π · u³)``
            derived from the large-argument approximation
            ``J₁(u) ~ √(2/(πu)) · cos(u − 3π/4)``.
        **kwargs
            Mounting keyword arguments forwarded to :class:`SymmetricAntenna`.
        """
        from scipy.special import j1

        _C = 299_792_458.0
        lam = _C / float(frequency)
        D = float(diameter)
        eta = float(eff)

        g_peak_lin = eta * (np.pi * D / lam) ** 2
        g_peak_dbi = 10.0 * np.log10(g_peak_lin)

        # Adaptive sampling: ≥20 points per first-null width
        sin_null = min(1.0, 1.22 * lam / D)
        theta_null = np.arcsin(sin_null)
        n_pts = max(500, int(np.ceil((np.pi / 2.0) / (theta_null / 20.0))))

        thetas = np.linspace(0.0, np.pi / 2.0, n_pts)
        u = np.pi * D * np.sin(thetas) / lam

        # [2 J₁(u)/u]²  with limit 1 at u=0
        u_safe = np.where(u < 1e-12, 1e-12, u)
        j1u = np.where(u < 1e-12, 1.0, 2.0 * j1(u_safe) / u_safe)
        f = j1u**2

        if envelope:
            # Beyond the first zero of J₁ (u ≈ 3.8317), replace with asymptotic envelope
            first_zero = 3.8317
            beyond = u > first_zero
            f_env = np.where(u > 0, 8.0 / (np.pi * np.maximum(u, 1e-30) ** 3), 1.0)
            f = np.where(beyond, f_env, f)

        g_lin = g_peak_lin * f
        gains = np.maximum(10.0 * np.log10(np.maximum(g_lin, 1e-20)), -60.0)

        # Extend to 180° with low gain (dish back-lobe region)
        angles_out = np.concatenate([np.degrees(thetas), [90.0, 180.0]])
        gains_out = np.concatenate([gains, [-60.0, -60.0]])

        # Remove duplicates at 90°
        _, unique_idx = np.unique(angles_out, return_index=True)
        angles_out = angles_out[unique_idx]
        gains_out = gains_out[unique_idx]

        return cls(angles_out, gains_out, **kwargs)

    @classmethod
    def from_s465(
        cls,
        diameter: float,
        frequency: float,
        main_lobe_model: bool = False,
        gmax_dbi: float | None = None,
        **kwargs,
    ) -> "SymmetricAntenna":
        """ITU-R S.465 reference Earth station antenna pattern.

        Parameters
        ----------
        diameter : float
            Reflector diameter (m).
        frequency : float
            Centre frequency (Hz).  Valid range per the Recommendation:
            2–31 GHz.
        main_lobe_model : bool, optional
            If *False* (default), the canonical ITU-R S.465-6 sidelobe
            envelope is used; the inner region (0° to φ_min) is completed
            with a flat plateau at *G*_max.  If *True*, the smooth
            parabolic main-lobe extension from APEREC013V01 is used,
            producing a continuous pattern from boresight to 180°.
        gmax_dbi : float, optional
            Peak on-axis gain (dBi).  When provided, overrides the value
            computed from *diameter* and *frequency* (which assumes η = 0.7).
            This matches the behaviour of tools such as ANSYS STK that
            accept *G*_max as a direct input.
        **kwargs
            Mounting keyword arguments forwarded to
            :class:`SymmetricAntenna`.

        References
        ----------
        .. [1] ITU Radiocommunication Assembly, "Reference radiation
               pattern of earth station antennas in the fixed-satellite
               service for use in coordination and interference assessment
               in the frequency range from 2 to 31 GHz," Recommendation
               ITU-R S.465-6, Jan. 2010.  (RRECS.4656201001IPDFE)
        .. [2] ITU-R BR Software, "Recommendation ITU-R S.465-5 reference
               Earth station antenna pattern for earth stations coordinated
               after 1993 in the frequency range from 2 to about 30 GHz,"
               APEREC013V01, Apr. 2022.
        """
        _C = 299_792_458.0
        D = float(diameter)
        lam = _C / float(frequency)
        dl = D / lam  # D/λ

        if gmax_dbi is not None:
            g_peak_dbi = float(gmax_dbi)
        else:
            eta = 0.7  # BR-software default efficiency
            g_peak_dbi = 10.0 * np.log10(eta * (np.pi * dl) ** 2)

        if main_lobe_model:
            # APEREC013V01 smooth main-lobe extension
            g1 = 32.0 if dl > 100 else -18.0 + 25.0 * np.log10(dl)
            phi_r = 1.0 if dl > 100 else 100.0 / dl  # degrees
            phi_m = (20.0 / dl) * np.sqrt(max(g_peak_dbi - g1, 0.0))
            phi_b = 10.0 ** (42.0 / 25.0)  # ≈ 48.0°

            # Parabolic main lobe: 0 → φ_m
            phi_ml = np.linspace(0.0, phi_m, 200, endpoint=False)
            g_ml = g_peak_dbi - 2.5e-3 * (dl * phi_ml) ** 2
            parts_a: list[npt.NDArray[np.floating]] = [phi_ml]
            parts_g: list[npt.NDArray[np.floating]] = [g_ml]

            # Transition plateau: φ_m → φ_r  (may be zero-width)
            if phi_r > phi_m:
                parts_a.append(np.array([phi_m, phi_r]))
                parts_g.append(np.array([g1, g1]))

            # Sidelobe envelope: φ_r → φ_b
            phi_sl = np.linspace(phi_r, phi_b, 300, endpoint=False)
            parts_a.append(phi_sl)
            parts_g.append(32.0 - 25.0 * np.log10(np.maximum(phi_sl, 1e-10)))

            # Far sidelobe: φ_b → 180°
            parts_a.append(np.array([phi_b, 180.0]))
            parts_g.append(np.array([-10.0, -10.0]))

        else:
            # Canonical S.465-6 sidelobe envelope
            if dl >= 50:
                phi_min = max(1.0, 100.0 / dl)
            else:
                phi_min = max(2.0, 114.0 * (lam / D) ** 1.09)

            phi_b = 48.0  # degrees

            # Flat main lobe: 0 → φ_min
            parts_a = [np.array([0.0, phi_min])]
            parts_g = [np.array([g_peak_dbi, g_peak_dbi])]

            # Sidelobe envelope: φ_min → 48°
            phi_sl = np.linspace(phi_min, phi_b, 500, endpoint=False)
            parts_a.append(phi_sl)
            parts_g.append(32.0 - 25.0 * np.log10(phi_sl))

            # Far sidelobe: 48° → 180°
            parts_a.append(np.array([phi_b, 180.0]))
            parts_g.append(np.array([-10.0, -10.0]))

        angles_out = np.concatenate(parts_a)
        gains_out = np.concatenate(parts_g)

        _, uniq = np.unique(angles_out, return_index=True)
        return cls(angles_out[uniq], gains_out[uniq], **kwargs)

    def _pattern_gain(
        self,
        off_boresight_rad: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        return np.interp(
            off_boresight_rad,
            self._angles_rad,
            self._gains_dbi,
        )
