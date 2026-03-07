"""RF link budget analysis."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .antenna import AbstractAntenna
from ..orbit.propagation import propagate_analytical
from ..orbit.frames import geodetic_to_ecef, ecef_to_eci, eci_to_ecef
from ..orbit.constants import EARTH_SEMI_MAJOR_AXIS

_C = 299_792_458.0          # speed of light (m/s)
_K_DB = 10.0 * np.log10(1.380649e-23)  # Boltzmann constant in dBW/K/Hz ≈ -228.6


class Link:
    """RF communication link between two antennas.

    Computes link margin at arbitrary times, accounting for free-space path
    loss, antenna pointing gain, central-body obstruction, and optionally
    ITU-R P.618 tropospheric attenuation for Spacecraft–GroundStation paths.

    Parameters
    ----------
    tx : AbstractAntenna
        Transmit antenna (must be attached to a host).
    rx : AbstractAntenna
        Receive antenna (must be attached to a host).
    tx_power_dbw : float
        Transmit power (dBW).
    frequency_hz : float
        Centre frequency (Hz).
    data_rate_bps : float
        Data rate (bit/s).
    rx_gt_db_k : float
        Receive system G/T at boresight (dB/K).  Off-boresight pointing loss
        from the *rx* antenna pattern is applied on top of this value.
    required_eb_n0_db : float
        Required Eb/N0 (dB).
    implementation_loss_db : float, optional
        Implementation loss (dB).  Default 2.
    misc_losses_db : float, optional
        Miscellaneous fixed losses (dB).  Default 0.
    use_p618 : bool, optional
        Whether to apply ITU-R P.618 atmospheric attenuation.  Only
        effective for Spacecraft–GroundStation links.  Default True.
    """

    def __init__(
        self,
        tx: AbstractAntenna,
        rx: AbstractAntenna,
        tx_power_dbw: float,
        frequency_hz: float,
        data_rate_bps: float,
        rx_gt_db_k: float,
        required_eb_n0_db: float,
        implementation_loss_db: float = 2.0,
        misc_losses_db: float = 0.0,
        use_p618: bool = True,
    ) -> None:
        from ..spacecraft import Spacecraft
        from ..ground_station import GroundStation

        if not isinstance(tx, AbstractAntenna):
            raise TypeError(
                f"tx must be an AbstractAntenna, got {type(tx).__name__!r}"
            )
        if not isinstance(rx, AbstractAntenna):
            raise TypeError(
                f"rx must be an AbstractAntenna, got {type(rx).__name__!r}"
            )
        if tx.host is None:
            raise ValueError("tx antenna must be attached to a host before use.")
        if rx.host is None:
            raise ValueError("rx antenna must be attached to a host before use.")
        if frequency_hz <= 0:
            raise ValueError(f"frequency_hz must be positive, got {frequency_hz}")
        if data_rate_bps <= 0:
            raise ValueError(f"data_rate_bps must be positive, got {data_rate_bps}")

        self._tx = tx
        self._rx = rx
        self._tx_power_dbw = float(tx_power_dbw)
        self._frequency_hz = float(frequency_hz)
        self._data_rate_bps = float(data_rate_bps)
        self._rx_gt_db_k = float(rx_gt_db_k)
        self._required_eb_n0_db = float(required_eb_n0_db)
        self._implementation_loss_db = float(implementation_loss_db)
        self._misc_losses_db = float(misc_losses_db)
        self._use_p618 = bool(use_p618)

        # Identify link type
        tx_host = tx.host
        rx_host = rx.host
        tx_is_sc = isinstance(tx_host, Spacecraft)
        rx_is_sc = isinstance(rx_host, Spacecraft)
        tx_is_gs = isinstance(tx_host, GroundStation)
        rx_is_gs = isinstance(rx_host, GroundStation)

        self._sc_gs_link = (tx_is_sc and rx_is_gs) or (tx_is_gs and rx_is_sc)
        self._gs = rx_host if rx_is_gs else (tx_host if tx_is_gs else None)

        # Central body radius for obstruction check
        sc_host = (tx_host if tx_is_sc else None) or (rx_host if rx_is_sc else None)
        self._body_radius = (
            sc_host.central_body_radius if sc_host is not None
            else EARTH_SEMI_MAJOR_AXIS
        )

    # --- properties ---

    @property
    def tx(self) -> AbstractAntenna:
        """Transmit antenna."""
        return self._tx

    @property
    def rx(self) -> AbstractAntenna:
        """Receive antenna."""
        return self._rx

    @property
    def tx_power_dbw(self) -> float:
        return self._tx_power_dbw

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def data_rate_bps(self) -> float:
        return self._data_rate_bps

    @property
    def rx_gt_db_k(self) -> float:
        return self._rx_gt_db_k

    @property
    def required_eb_n0_db(self) -> float:
        return self._required_eb_n0_db

    @property
    def implementation_loss_db(self) -> float:
        return self._implementation_loss_db

    @property
    def misc_losses_db(self) -> float:
        return self._misc_losses_db

    @property
    def use_p618(self) -> bool:
        return self._use_p618

    # --- public API ---

    def link_margin(
        self,
        t: npt.ArrayLike,
        *,
        availability_pct: float = 99.9,
    ) -> npt.NDArray[np.floating]:
        """Compute link margin at the given times.

        Parameters
        ----------
        t : array_like of datetime64[us], shape (N,) or scalar
            Evaluation timestamps.
        availability_pct : float, optional
            Link availability target (%) used for ITU-R P.618 atmospheric
            attenuation.  The attenuation not exceeded for this percentage
            of the time is applied.  Default 99.9.

        Returns
        -------
        ndarray, shape (N,) or scalar float
            Link margin (dB) at each timestamp.  Timesteps where the central
            body blocks the line of sight are returned as NaN.
        """
        scalar_input = np.asarray(t).ndim == 0
        t_arr = np.atleast_1d(np.asarray(t, dtype='datetime64[us]'))

        r_tx, v_tx = self._host_eci(self._tx.host, t_arr)
        r_rx, v_rx = self._host_eci(self._rx.host, t_arr)

        # Obstruction
        blocked = self._los_blocked(r_tx, r_rx)

        # Range and FSPL
        delta = r_rx - r_tx  # (N,3)
        range_m = np.linalg.norm(delta, axis=1)  # (N,)
        fspl_db = 20.0 * np.log10(
            4.0 * np.pi * range_m * self._frequency_hz / _C
        )

        # Tx gain
        g_tx = self._tx.gain(t_arr, delta, frame='eci', r_eci=r_tx, v_eci=v_tx)

        # Rx gain and pointing loss
        g_rx = self._rx.gain(t_arr, -delta, frame='eci', r_eci=r_rx, v_eci=v_rx)
        pointing_loss = g_rx - self._rx.peak_gain_dbi  # ≤ 0

        # C/N0 (dBHz)
        c_n0 = (
            self._tx_power_dbw
            + g_tx
            - fspl_db
            + self._rx_gt_db_k
            + pointing_loss
            - _K_DB  # subtracting negative Boltzmann → adding 228.6
        )

        # Eb/N0
        eb_n0 = c_n0 - 10.0 * np.log10(self._data_rate_bps)

        # Atmospheric attenuation
        atm_loss = self._p618_attenuation(r_tx, r_rx, t_arr, availability_pct)

        # Margin
        margin = (
            eb_n0
            - self._required_eb_n0_db
            - self._implementation_loss_db
            - self._misc_losses_db
            - atm_loss
        )
        margin[blocked] = np.nan

        if scalar_input:
            return float(margin[0])
        return margin

    # --- internal helpers ---

    @staticmethod
    def _host_eci(
        host,
        t_arr: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Return (r_eci, v_eci) shape (N,3) for a Spacecraft or GroundStation."""
        from ..spacecraft import Spacecraft
        from ..ground_station import GroundStation

        if isinstance(host, Spacecraft):
            r, v = propagate_analytical(
                t_arr, **host.keplerian_params, propagator_type=host.propagator_type
            )
            return r, v
        elif isinstance(host, GroundStation):
            r_ecef = geodetic_to_ecef(
                np.radians(host.lat), np.radians(host.lon), host.alt
            )
            n = len(t_arr)
            r_ecef_tiled = np.tile(r_ecef, (n, 1))
            r_eci = ecef_to_eci(r_ecef_tiled, t_arr)
            v_eci = np.zeros((n, 3))
            return r_eci, v_eci
        else:
            raise TypeError(f"Unsupported host type: {type(host).__name__!r}")

    def _los_blocked(
        self,
        r_tx: npt.NDArray,
        r_rx: npt.NDArray,
    ) -> npt.NDArray[np.bool_]:
        """Return boolean array; True where the central body blocks LOS.

        The blocking sphere radius is capped to just below the minimum
        geocentric distance of either endpoint.  This is necessary because
        the Earth is oblate: using the equatorial radius as a sphere would
        place high-latitude ground stations "inside" the sphere and
        incorrectly flag all links as blocked.
        """
        d = r_rx - r_tx  # (N,3)
        r_tx_dot_d = np.einsum('ij,ij->i', r_tx, d)
        d_dot_d = np.einsum('ij,ij->i', d, d)
        t_star = np.clip(-r_tx_dot_d / d_dot_d, 0.0, 1.0)
        closest = r_tx + t_star[:, np.newaxis] * d
        min_dist_sq = np.einsum('ij,ij->i', closest, closest)

        # Effective blocking radius: no larger than the smaller of the two
        # endpoint distances (so neither endpoint is ever "inside" the sphere).
        r_tx_sq = np.einsum('ij,ij->i', r_tx, r_tx)
        r_rx_sq = np.einsum('ij,ij->i', r_rx, r_rx)
        block_sq = np.minimum(
            self._body_radius ** 2,
            np.minimum(r_tx_sq, r_rx_sq) * (1.0 - 1e-6),
        )
        return min_dist_sq < block_sq

    def _p618_attenuation(
        self,
        r_tx: npt.NDArray,
        r_rx: npt.NDArray,
        t_arr: npt.NDArray,
        availability_pct: float,
    ) -> npt.NDArray[np.floating]:
        """Return P.618 atmospheric attenuation (dB), shape (N,).

        Returns zeros if P.618 is disabled or the link is not SC–GS.
        """
        n = len(t_arr)
        if not self._use_p618 or not self._sc_gs_link:
            return np.zeros(n)

        from ..spacecraft import Spacecraft
        from ..ground_station import GroundStation

        gs = self._gs
        # Identify which side is the spacecraft
        if isinstance(self._tx.host, Spacecraft):
            r_sc_eci = r_tx
        else:
            r_sc_eci = r_rx

        # Elevation angle from GS to SC
        r_sc_ecef = eci_to_ecef(r_sc_eci, t_arr)
        r_gs_ecef = geodetic_to_ecef(
            np.radians(gs.lat), np.radians(gs.lon), gs.alt
        )
        delta_ecef = r_sc_ecef - r_gs_ecef  # (N,3)

        lat_rad = np.radians(gs.lat)
        lon_rad = np.radians(gs.lon)
        up = np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ])
        range_m = np.linalg.norm(delta_ecef, axis=1)
        up_component = delta_ecef @ up
        elevation_deg = np.degrees(np.arcsin(
            np.clip(up_component / range_m, -1.0, 1.0)
        ))
        elevation_deg = np.maximum(elevation_deg, 0.1)  # itur undefined at 0°

        import itur
        freq_ghz = self._frequency_hz / 1e9
        p = 100.0 - availability_pct  # % of time exceeded

        atm_loss = np.empty(n)
        for k in range(n):
            result = itur.atmospheric_attenuation_slant_path(
                gs.lat, gs.lon, freq_ghz, elevation_deg[k], p, D=0
            )
            val = result.value if hasattr(result, 'value') else float(result)
            atm_loss[k] = float(np.asarray(val))

        return atm_loss
