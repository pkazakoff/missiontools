"""Interference analysis between space networks."""

from __future__ import annotations

from itertools import product

import numpy as np
import numpy.typing as npt

from .antenna import AbstractAntenna
from ..condition.condition import AbstractCondition
from ..orbit.access import (
    earth_access_intervals,
    space_to_space_access,
    space_to_space_access_intervals,
)
from ..orbit.constants import EARTH_MEAN_RADIUS
from ..orbit.frames import geodetic_to_ecef, ecef_to_eci
from ..orbit.utils import host_eci_state

_C = 299_792_458.0


def _host_eci(
    host,
    t_arr: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    return host_eci_state(host, t_arr)


def _get_host_type(host):
    from ..spacecraft import Spacecraft
    from ..ground_station import GroundStation

    if isinstance(host, Spacecraft):
        return "sc"
    elif isinstance(host, GroundStation):
        return "gs"
    raise TypeError(f"Unsupported host type: {type(host).__name__!r}")


def _gs_gs_access_intervals(
    gs_a,
    gs_b,
    t_start: np.datetime64,
    t_end: np.datetime64,
) -> list[tuple[np.datetime64, np.datetime64]]:
    r_a_ecef = geodetic_to_ecef(np.radians(gs_a.lat), np.radians(gs_a.lon), gs_a.alt)
    r_b_ecef = geodetic_to_ecef(np.radians(gs_b.lat), np.radians(gs_b.lon), gs_b.alt)
    clear = bool(
        space_to_space_access(
            r_a_ecef.reshape(1, 3),
            r_b_ecef.reshape(1, 3),
            EARTH_MEAN_RADIUS,
        )[0]
    )
    if clear:
        return [(t_start, t_end)]
    return []


def _get_access_intervals(
    cache: dict,
    host_a,
    host_b,
    t_start: np.datetime64,
    t_end: np.datetime64,
    max_step: np.timedelta64,
) -> list[tuple[np.datetime64, np.datetime64]]:
    key = (id(host_a), id(host_b))
    if key in cache:
        return cache[key]

    type_a = _get_host_type(host_a)
    type_b = _get_host_type(host_b)

    if type_a == "sc" and type_b == "sc":
        intervals = space_to_space_access_intervals(
            t_start=t_start,
            t_end=t_end,
            keplerian_params_1=host_a.keplerian_params,
            keplerian_params_2=host_b.keplerian_params,
            body_radius=host_a.central_body_radius,
            propagator_type_1=host_a.propagator_type,
            propagator_type_2=host_b.propagator_type,
            max_step=max_step,
        )
    elif type_a == "sc" and type_b == "gs":
        intervals = earth_access_intervals(
            t_start=t_start,
            t_end=t_end,
            keplerian_params=host_a.keplerian_params,
            lat=np.radians(host_b.lat),
            lon=np.radians(host_b.lon),
            alt=host_b.alt,
            el_min=0.0,
            propagator_type=host_a.propagator_type,
            max_step=max_step,
        )
    elif type_a == "gs" and type_b == "sc":
        intervals = earth_access_intervals(
            t_start=t_start,
            t_end=t_end,
            keplerian_params=host_b.keplerian_params,
            lat=np.radians(host_a.lat),
            lon=np.radians(host_a.lon),
            alt=host_a.alt,
            el_min=0.0,
            propagator_type=host_b.propagator_type,
            max_step=max_step,
        )
    elif type_a == "gs" and type_b == "gs":
        intervals = _gs_gs_access_intervals(host_a, host_b, t_start, t_end)
    else:
        intervals = []

    cache[key] = intervals
    return intervals


def _intersect_intervals(
    intervals_a: list[tuple[np.datetime64, np.datetime64]],
    intervals_b: list[tuple[np.datetime64, np.datetime64]],
) -> list[tuple[np.datetime64, np.datetime64]]:
    result = []
    i, j = 0, 0
    while i < len(intervals_a) and j < len(intervals_b):
        start = max(intervals_a[i][0], intervals_b[j][0])
        end = min(intervals_a[i][1], intervals_b[j][1])
        if start < end:
            result.append((start, end))
        if intervals_a[i][1] < intervals_b[j][1]:
            i += 1
        else:
            j += 1
    return result


def _union_intervals(
    intervals: list[tuple[np.datetime64, np.datetime64]],
) -> list[tuple[np.datetime64, np.datetime64]]:
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda iv: iv[0])
    merged = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _condition_filter_intervals(
    intervals: list[tuple[np.datetime64, np.datetime64]],
    event_step_td: np.timedelta64,
    *conditions: AbstractCondition | None,
) -> list[tuple[np.datetime64, np.datetime64]]:
    filtered: list[tuple[np.datetime64, np.datetime64]] = []
    for iv_start, iv_end in intervals:
        duration_us = int((iv_end - iv_start) / np.timedelta64(1, "us"))
        step_us = int(event_step_td / np.timedelta64(1, "us"))
        if step_us <= 0:
            step_us = 1
        n_samples = duration_us // step_us + 1
        offsets_us = np.arange(n_samples, dtype=np.int64) * step_us
        sample_times = iv_start + offsets_us.astype("timedelta64[us]")
        if offsets_us[-1] < duration_us:
            sample_times = np.append(
                sample_times,
                iv_start + np.timedelta64(duration_us, "us"),
            )
        mask = np.ones(len(sample_times), dtype=bool)
        for cond in conditions:
            if cond is not None:
                mask &= cond.at(sample_times)
        runs = _find_exceedance_runs(mask.astype(np.float64), 0.5)
        for r_start, r_end in runs:
            filtered.append((sample_times[r_start], sample_times[r_end]))
    return filtered


def _total_interval_seconds(
    intervals: list[tuple[np.datetime64, np.datetime64]],
) -> float:
    total = 0.0
    for start, end in intervals:
        total += float((end - start) / np.timedelta64(1, "s"))
    return total


def _find_exceedance_runs(
    psd: npt.NDArray[np.floating],
    threshold: float,
) -> list[tuple[int, int]]:
    above = psd >= threshold
    if not np.any(above):
        return []

    runs = []
    in_run = False
    start = 0
    for k in range(len(above)):
        if above[k] and not in_run:
            start = k
            in_run = True
        elif not above[k] and in_run:
            runs.append((start, k - 1))
            in_run = False
    if in_run:
        runs.append((start, len(above) - 1))
    return runs


class InterferenceAnalysis:
    """Analyze interference risk between space networks.

    Parameters
    ----------
    f_MHz : float
        Centre frequency of the channel in MHz.
    """

    def __init__(self, f_MHz: float) -> None:
        if f_MHz <= 0:
            raise ValueError(f"f_MHz must be positive, got {f_MHz}")
        self._f_Hz = float(f_MHz) * 1e6
        self._victim_txs: list[dict] = []
        self._victim_rxs: list[dict] = []
        self._interfering_txs: list[dict] = []
        self._cached_events: list[dict] | None = None
        self._cached_access_intervals: dict[str, dict[str, list]] | None = None
        self._cached_access_totals: dict[str, dict[str, float]] | None = None
        self._compute_psd_threshold: float | None = None

    def _invalidate_cache(self) -> None:
        self._cached_events = None
        self._cached_access_intervals = None
        self._cached_access_totals = None
        self._compute_psd_threshold = None

    def add_victim_tx(
        self,
        name: str,
        antenna: AbstractAntenna,
        tx_psd: float,
        condition: AbstractCondition | None = None,
    ) -> None:
        if not isinstance(antenna, AbstractAntenna):
            raise TypeError(
                f"antenna must be an AbstractAntenna, got {type(antenna).__name__!r}"
            )
        if antenna.host is None:
            raise ValueError("antenna must be attached to a host before use.")
        if condition is not None and not isinstance(condition, AbstractCondition):
            raise TypeError(
                f"condition must be an AbstractCondition or None, "
                f"got {type(condition).__name__!r}"
            )
        self._victim_txs.append(
            {
                "name": name,
                "antenna": antenna,
                "tx_psd": float(tx_psd),
                "condition": condition,
            }
        )
        self._invalidate_cache()

    def add_victim_rx(
        self,
        name: str,
        antenna: AbstractAntenna,
        condition: AbstractCondition | None = None,
    ) -> None:
        if not isinstance(antenna, AbstractAntenna):
            raise TypeError(
                f"antenna must be an AbstractAntenna, got {type(antenna).__name__!r}"
            )
        if antenna.host is None:
            raise ValueError("antenna must be attached to a host before use.")
        if condition is not None and not isinstance(condition, AbstractCondition):
            raise TypeError(
                f"condition must be an AbstractCondition or None, "
                f"got {type(condition).__name__!r}"
            )
        self._victim_rxs.append(
            {
                "name": name,
                "antenna": antenna,
                "condition": condition,
            }
        )
        self._invalidate_cache()

    def add_interfering_tx(
        self,
        name: str,
        antenna: AbstractAntenna,
        tx_psd: float,
        condition: AbstractCondition | None = None,
    ) -> None:
        if not isinstance(antenna, AbstractAntenna):
            raise TypeError(
                f"antenna must be an AbstractAntenna, got {type(antenna).__name__!r}"
            )
        if antenna.host is None:
            raise ValueError("antenna must be attached to a host before use.")
        if condition is not None and not isinstance(condition, AbstractCondition):
            raise TypeError(
                f"condition must be an AbstractCondition or None, "
                f"got {type(condition).__name__!r}"
            )
        self._interfering_txs.append(
            {
                "name": name,
                "antenna": antenna,
                "tx_psd": float(tx_psd),
                "condition": condition,
            }
        )
        self._invalidate_cache()

    def compute(
        self,
        psd_threshold: float,
        start_time: npt.ArrayLike,
        end_time: npt.ArrayLike,
        max_step: float = 10.0,
        event_step: float = 1.0,
    ) -> tuple[list[dict], dict[str, dict[str, float]]]:
        """Compute interference events.

        Parameters
        ----------
        psd_threshold : float
            Interference PSD threshold in dBW/Hz.
        start_time : array_like of datetime64
            Start of the analysis window.
        end_time : array_like of datetime64
            End of the analysis window.
        max_step : float, optional
            Coarse scan step for access interval detection in seconds.
            Default 10.0.
        event_step : float, optional
            Sampling step within candidate intervals in seconds.
            Default 1.0.

        Returns
        -------
        events : list[dict]
            Each dict represents an interference event with keys:
            ``victim_tx``, ``victim_rx``, ``interfering_tx`` (str),
            ``start_time``, ``end_time`` (datetime64),
            ``max_interferer_psd`` (float, dBW/Hz),
            ``times`` (ndarray of datetime64),
            ``interferer_psd``, ``victim_psd`` (ndarray, dBW/Hz).
        access_totals : dict[str, dict[str, float]]
            Nested dictionary of total pairwise condition-filtered
            access times in seconds.  Index as
            ``access_totals[victim_tx_name][victim_rx_name]``.
            Access intervals are filtered by the conditions applied
            to the victim transmitter and receiver.
        """
        if not self._victim_txs:
            raise ValueError("No victim transmitters have been added.")
        if not self._victim_rxs:
            raise ValueError("No victim receivers have been added.")
        if not self._interfering_txs:
            raise ValueError("No interfering transmitters have been added.")

        t_start = np.asarray(start_time, dtype="datetime64[us]")
        t_end = np.asarray(end_time, dtype="datetime64[us]")
        max_step_td = np.timedelta64(int(round(max_step * 1e6)), "us")
        event_step_td = np.timedelta64(int(round(event_step * 1e6)), "us")

        access_cache: dict = {}
        events: list[dict] = []
        access_intervals: dict[str, dict[str, list]] = {}

        for vtx, vrx, itx in product(
            self._victim_txs, self._victim_rxs, self._interfering_txs
        ):
            vtx_name = vtx["name"]
            vrx_name = vrx["name"]

            if vtx_name not in access_intervals:
                access_intervals[vtx_name] = {}
            if vrx_name not in access_intervals[vtx_name]:
                vtx_host = vtx["antenna"].host
                vrx_host = vrx["antenna"].host
                geom_intervals = _get_access_intervals(
                    access_cache,
                    vtx_host,
                    vrx_host,
                    t_start,
                    t_end,
                    max_step_td,
                )
                access_intervals[vtx_name][vrx_name] = _condition_filter_intervals(
                    geom_intervals,
                    event_step_td,
                    vtx["condition"],
                    vrx["condition"],
                )

            vtx_host = vtx["antenna"].host
            vrx_host = vrx["antenna"].host
            itx_host = itx["antenna"].host

            victim_intervals = _get_access_intervals(
                access_cache,
                vtx_host,
                vrx_host,
                t_start,
                t_end,
                max_step_td,
            )
            interferer_intervals = _get_access_intervals(
                access_cache,
                itx_host,
                vrx_host,
                t_start,
                t_end,
                max_step_td,
            )

            candidate_intervals = _intersect_intervals(
                victim_intervals,
                interferer_intervals,
            )

            for ci_start, ci_end in candidate_intervals:
                duration_us = int((ci_end - ci_start) / np.timedelta64(1, "us"))
                step_us = int(event_step_td / np.timedelta64(1, "us"))
                if step_us <= 0:
                    step_us = 1

                n_samples = duration_us // step_us + 1
                offsets_us = np.arange(n_samples, dtype=np.int64) * step_us
                sample_times = ci_start + offsets_us.astype("timedelta64[us]")

                if offsets_us[-1] < duration_us:
                    sample_times = np.append(
                        sample_times,
                        ci_start + np.timedelta64(duration_us, "us"),
                    )

                mask = np.ones(len(sample_times), dtype=bool)
                if vtx["condition"] is not None:
                    mask &= vtx["condition"].at(sample_times)
                if vrx["condition"] is not None:
                    mask &= vrx["condition"].at(sample_times)
                if itx["condition"] is not None:
                    mask &= itx["condition"].at(sample_times)

                if not np.any(mask):
                    continue

                times_masked = sample_times[mask]

                r_vtx, v_vtx = _host_eci(vtx_host, times_masked)
                r_vrx, v_vrx = _host_eci(vrx_host, times_masked)
                r_itx, v_itx = _host_eci(itx_host, times_masked)

                delta_v_vrx = r_vrx - r_vtx
                range_v = np.linalg.norm(delta_v_vrx, axis=1)
                fspl_v = 20.0 * np.log10(4.0 * np.pi * range_v * self._f_Hz / _C)

                delta_i_vrx = r_vrx - r_itx
                range_i = np.linalg.norm(delta_i_vrx, axis=1)
                fspl_i = 20.0 * np.log10(4.0 * np.pi * range_i * self._f_Hz / _C)

                g_vtx = vtx["antenna"].gain(
                    times_masked,
                    delta_v_vrx,
                    frame="eci",
                    r_eci=r_vtx,
                    v_eci=v_vtx,
                )
                g_vrx_v = vrx["antenna"].gain(
                    times_masked,
                    -delta_v_vrx,
                    frame="eci",
                    r_eci=r_vrx,
                    v_eci=v_vrx,
                )

                g_itx = itx["antenna"].gain(
                    times_masked,
                    delta_i_vrx,
                    frame="eci",
                    r_eci=r_itx,
                    v_eci=v_itx,
                )
                g_vrx_i = vrx["antenna"].gain(
                    times_masked,
                    -delta_i_vrx,
                    frame="eci",
                    r_eci=r_vrx,
                    v_eci=v_vrx,
                )

                victim_psd = vtx["tx_psd"] + g_vtx - fspl_v + g_vrx_v
                interf_psd = itx["tx_psd"] + g_itx - fspl_i + g_vrx_i

                runs = _find_exceedance_runs(interf_psd, psd_threshold)

                for run_start, run_end in runs:
                    events.append(
                        {
                            "victim_tx": vtx_name,
                            "victim_rx": vrx_name,
                            "interfering_tx": itx["name"],
                            "start_time": times_masked[run_start],
                            "end_time": times_masked[run_end],
                            "max_interferer_psd": float(
                                np.max(interf_psd[run_start : run_end + 1])
                            ),
                            "times": times_masked[run_start : run_end + 1].copy(),
                            "interferer_psd": interf_psd[
                                run_start : run_end + 1
                            ].copy(),
                            "victim_psd": victim_psd[run_start : run_end + 1].copy(),
                        }
                    )

        access_totals: dict[str, dict[str, float]] = {}
        for vtx_name, rx_map in access_intervals.items():
            access_totals[vtx_name] = {}
            for vrx_name, intervals in rx_map.items():
                access_totals[vtx_name][vrx_name] = _total_interval_seconds(intervals)

        self._cached_events = events
        self._cached_access_intervals = access_intervals
        self._cached_access_totals = access_totals
        self._compute_psd_threshold = psd_threshold

        return events, access_totals

    def interference_percentage(
        self,
        psd_threshold: float | list[float] | npt.ArrayLike,
        victim_tx: str | list[str] | npt.ArrayLike,
        victim_rx: str,
        interfering_tx: str | list[str] | npt.ArrayLike,
    ) -> float | npt.NDArray[np.floating]:
        """Compute interference percentage from the last ``compute()`` results.

        Interference percentage is defined as::

            100.0 * T_interf / T_access

        where *T_interf* is the total time (in seconds) during which
        the received interferer PSD at *victim_rx* is at or above
        *psd_threshold* while **at least one** specified *victim_tx*
        and **at least one** specified *interfering_tx* are visible
        (condition-filtered), and *T_access* is the total
        condition-filtered time that at least one specified
        *victim_tx* is visible to *victim_rx*.

        Parameters
        ----------
        psd_threshold : float or list[float] or array_like
            Interference PSD threshold(s) in dBW/Hz.  All values
            must be greater than or equal to the threshold passed to
            the most recent ``compute()`` call.  When an array is
            provided, an array of percentages is returned (one per
            threshold), suitable for generating cumulative
            interference plots.
        victim_tx : str or list[str] or array_like
            Name(s) of the victim transmitter(s) to include.
        victim_rx : str
            Name of the single victim receiver.
        interfering_tx : str or list[str] or array_like
            Name(s) of the interfering transmitter(s) to include.

        Returns
        -------
        float or ndarray
            Interference percentage(s) in the range [0, 100].
            Returns a scalar ``float`` when *psd_threshold* is a
            scalar, or an ``ndarray`` when *psd_threshold* is a
            sequence.  Returns 0.0 (or an array of zeros) when the
            denominator (total access time) is zero.

        Raises
        ------
        RuntimeError
            If ``compute()`` has not been run or its results have been
            invalidated.
        ValueError
            If any *psd_threshold* value is less than the threshold
            used in the most recent ``compute()`` call.
        KeyError
            If a specified transmitter or receiver name is not found
            in the cached results.
        """
        if self._cached_events is None:
            raise RuntimeError("No cached results.  Call compute() first.")

        thresholds = np.atleast_1d(np.asarray(psd_threshold, dtype=float))
        if np.any(thresholds < self._compute_psd_threshold):
            raise ValueError(
                f"all psd_threshold values must be >= the threshold "
                f"used in compute() ({self._compute_psd_threshold})"
            )

        if isinstance(victim_tx, str):
            vtx_names = [victim_tx]
        else:
            vtx_names = list(victim_tx)

        if isinstance(interfering_tx, str):
            itx_names = [interfering_tx]
        else:
            itx_names = list(interfering_tx)

        vtx_set = set(vtx_names)
        itx_set = set(itx_names)

        assert self._cached_access_intervals is not None

        denom_intervals: list[tuple[np.datetime64, np.datetime64]] = []
        for vtx_name in vtx_names:
            rx_map = self._cached_access_intervals.get(vtx_name)
            if rx_map is None:
                raise KeyError(
                    f"Victim transmitter {vtx_name!r} not found in cached results"
                )
            intervals = rx_map.get(victim_rx)
            if intervals is None:
                raise KeyError(
                    f"Victim receiver {victim_rx!r} not found for transmitter "
                    f"{vtx_name!r} in cached results"
                )
            denom_intervals.extend(intervals)
        denom_intervals = _union_intervals(denom_intervals)
        denominator = _total_interval_seconds(denom_intervals)

        if denominator == 0.0:
            result = np.zeros(len(thresholds), dtype=float)
            return (
                float(result[0])
                if result.ndim == 0 or len(result) == 1 and np.isscalar(psd_threshold)
                else result
            )

        matched_events = [
            ev
            for ev in self._cached_events
            if ev["victim_tx"] in vtx_set
            and ev["victim_rx"] == victim_rx
            and ev["interfering_tx"] in itx_set
        ]

        compute_threshold = self._compute_psd_threshold
        results = np.empty(len(thresholds), dtype=float)

        for idx, thresh in enumerate(thresholds):
            numer_intervals: list[tuple[np.datetime64, np.datetime64]] = []
            for event in matched_events:
                if thresh == compute_threshold:
                    numer_intervals.append((event["start_time"], event["end_time"]))
                else:
                    runs = _find_exceedance_runs(event["interferer_psd"], float(thresh))
                    evt_times = event["times"]
                    for r_start, r_end in runs:
                        numer_intervals.append((evt_times[r_start], evt_times[r_end]))
            numer_intervals = _union_intervals(numer_intervals)
            numerator = _total_interval_seconds(numer_intervals)
            results[idx] = 100.0 * numerator / denominator

        if np.isscalar(psd_threshold) and not isinstance(
            psd_threshold, (list, np.ndarray)
        ):
            return float(results[0])
        return results
