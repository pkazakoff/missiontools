from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from ..orbit.frames import (
    geodetic_to_ecef,
    eci_to_ecef,
    lvlh_to_eci,
    sun_vec_eci,
    geodetic_up,
)
from ..cache import cached_propagate_analytical


@dataclass
class CoverageConstraints:
    el_min: float | None
    sza_max: float | None
    sza_min: float | None
    sin_el_min: float
    cos_sza_max: float | None
    cos_sza_min: float | None

    @property
    def use_sza(self) -> bool:
        return self.sza_max is not None or self.sza_min is not None

    @classmethod
    def from_angles(
        cls,
        el_min: float,
        sza_max: float | None = None,
        sza_min: float | None = None,
    ) -> CoverageConstraints:
        return cls(
            el_min=el_min,
            sza_max=sza_max,
            sza_min=sza_min,
            sin_el_min=float(np.sin(el_min)),
            cos_sza_max=float(np.cos(sza_max)) if sza_max is not None else None,
            cos_sza_min=float(np.cos(sza_min)) if sza_min is not None else None,
        )


class SensorSpec(NamedTuple):
    keplerian_params: dict
    propagator_type: str
    fov_type: str
    pointing_lvlh: npt.NDArray | None
    cos_fov: float | None = None
    u1_lvlh: npt.NDArray | None = None
    u2_lvlh: npt.NDArray | None = None
    tan_theta1: float | None = None
    tan_theta2: float | None = None
    active_intervals: list[tuple[np.datetime64, np.datetime64]] | None = None


def make_sensor_spec(
    keplerian_params: dict,
    propagator_type: str,
    fov_pointing_lvlh: npt.NDArray | None,
    fov_half_angle: float | None,
) -> SensorSpec:
    _fov_given = (fov_pointing_lvlh is not None, fov_half_angle is not None)
    if any(_fov_given) and not all(_fov_given):
        raise ValueError(
            "fov_pointing_lvlh and fov_half_angle must both be provided together"
        )
    if fov_pointing_lvlh is not None and fov_half_angle is not None:
        pointing_norm = np.asarray(
            fov_pointing_lvlh, dtype=np.float64
        ) / np.linalg.norm(fov_pointing_lvlh)
        cos_fov = float(np.cos(fov_half_angle))
        return SensorSpec(
            keplerian_params, propagator_type, "conic", pointing_norm, cos_fov=cos_fov
        )
    return SensorSpec(keplerian_params, propagator_type, "none", None)


def make_sensor_spec_from_fov(
    keplerian_params: dict,
    propagator_type: str,
    fov: dict,
) -> SensorSpec:
    fov_type = fov["fov_type"]
    pointing = np.asarray(fov["pointing_lvlh"], dtype=np.float64)
    pointing = pointing / np.linalg.norm(pointing)

    if fov_type == "conic":
        return SensorSpec(
            keplerian_params,
            propagator_type,
            "conic",
            pointing,
            cos_fov=fov["cos_half_angle"],
        )

    if fov_type == "rectangular":
        u1 = np.asarray(fov["u1_lvlh"], dtype=np.float64)
        u2 = np.asarray(fov["u2_lvlh"], dtype=np.float64)
        return SensorSpec(
            keplerian_params,
            propagator_type,
            "rectangular",
            pointing,
            u1_lvlh=u1,
            u2_lvlh=u2,
            tan_theta1=fov["tan_theta1"],
            tan_theta2=fov["tan_theta2"],
        )

    raise ValueError(f"Unknown fov_type: {fov_type!r}")


def _build_gs(
    lat: npt.NDArray,
    lon: npt.NDArray,
    alt: float | npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    gs_ecef = geodetic_to_ecef(lat, lon, alt)
    up = geodetic_up(lat, lon)
    return gs_ecef, up


def _sun_ecef(t_arr: npt.NDArray) -> npt.NDArray:
    return eci_to_ecef(sun_vec_eci(t_arr), t_arr)


def _pointing_ecef(
    pointing_lvlh: npt.NDArray,
    r_eci: npt.NDArray,
    v_eci: npt.NDArray,
    t_arr: npt.NDArray,
) -> npt.NDArray:
    T = len(t_arr)
    p_eci = lvlh_to_eci(np.tile(pointing_lvlh, (T, 1)), r_eci, v_eci)
    return eci_to_ecef(p_eci, t_arr)


def _visibility(
    r_eci: npt.NDArray,
    t_arr: npt.NDArray,
    gs_ecef: npt.NDArray,
    up: npt.NDArray,
    sin_el_min: float,
    pointing_ecef: npt.NDArray | None = None,
    cos_fov: float | None = None,
    u1_ecef: npt.NDArray | None = None,
    u2_ecef: npt.NDArray | None = None,
    tan_theta1: float | None = None,
    tan_theta2: float | None = None,
    sun_ecef: npt.NDArray | None = None,
    cos_sza_max: float | None = None,
    cos_sza_min: float | None = None,
) -> npt.NDArray[np.bool_]:
    r_ecef = eci_to_ecef(r_eci, t_arr)
    los = r_ecef[:, np.newaxis, :] - gs_ecef[np.newaxis, :, :]
    norm = np.maximum(np.linalg.norm(los, axis=2, keepdims=True), 1e-10)
    los_hat = los / norm
    sin_el = np.einsum("tmi,mi->tm", los_hat, up)
    vis = sin_el >= sin_el_min

    if pointing_ecef is not None and cos_fov is not None:
        fov_cos = np.einsum("tmi,ti->tm", -los_hat, pointing_ecef)
        vis &= fov_cos >= cos_fov

    if pointing_ecef is not None and u1_ecef is not None:
        d_along = np.einsum("tmi,ti->tm", -los_hat, pointing_ecef)
        d_perp1 = np.einsum("tmi,ti->tm", -los_hat, u1_ecef)
        d_perp2 = np.einsum("tmi,ti->tm", -los_hat, u2_ecef)
        safe_along = np.where(d_along > 0, d_along, 1.0)
        vis &= d_along > 0
        vis &= np.abs(d_perp1) <= tan_theta1 * safe_along
        vis &= np.abs(d_perp2) <= tan_theta2 * safe_along

    if sun_ecef is not None:
        cos_sza = np.einsum("ti,mi->tm", sun_ecef, up)
        if cos_sza_max is not None:
            vis &= cos_sza >= cos_sza_max
        if cos_sza_min is not None:
            vis &= cos_sza <= cos_sza_min

    return vis


def _compute_vis_batch_multi(
    t_batch: npt.NDArray,
    sensor_specs: list,
    gs_ecef: npt.NDArray,
    up: npt.NDArray,
    constraints: CoverageConstraints,
) -> npt.NDArray[np.bool_]:
    T = len(t_batch)
    M = gs_ecef.shape[0]
    vis = np.zeros((T, M), dtype=np.bool_)

    for spec in sensor_specs:
        kp, prop_type, fov_type = (
            spec.keplerian_params,
            spec.propagator_type,
            spec.fov_type,
        )
        r, v = cached_propagate_analytical(t_batch, **kp, propagator_type=prop_type)

        if fov_type == "conic":
            pt_ecef = _pointing_ecef(spec.pointing_lvlh, r, v, t_batch)
            sensor_vis = _visibility(
                r,
                t_batch,
                gs_ecef,
                up,
                constraints.sin_el_min,
                pointing_ecef=pt_ecef,
                cos_fov=spec.cos_fov,
            )
        elif fov_type == "rectangular":
            pt_ecef = _pointing_ecef(spec.pointing_lvlh, r, v, t_batch)
            u1_ecef = _pointing_ecef(spec.u1_lvlh, r, v, t_batch)
            u2_ecef = _pointing_ecef(spec.u2_lvlh, r, v, t_batch)
            sensor_vis = _visibility(
                r,
                t_batch,
                gs_ecef,
                up,
                constraints.sin_el_min,
                pointing_ecef=pt_ecef,
                u1_ecef=u1_ecef,
                u2_ecef=u2_ecef,
                tan_theta1=spec.tan_theta1,
                tan_theta2=spec.tan_theta2,
            )
        else:
            sensor_vis = _visibility(r, t_batch, gs_ecef, up, constraints.sin_el_min)

        if spec.active_intervals is not None:
            active = np.zeros(T, dtype=np.bool_)
            for t0, t1 in spec.active_intervals:
                active |= (t_batch >= t0) & (t_batch <= t1)
            sensor_vis &= active[:, np.newaxis]

        vis |= sensor_vis

    if constraints.use_sza:
        sun_e = _sun_ecef(t_batch)
        cos_sza = np.einsum("ti,mi->tm", sun_e, up)
        if constraints.cos_sza_max is not None:
            vis &= cos_sza >= constraints.cos_sza_max
        if constraints.cos_sza_min is not None:
            vis &= cos_sza <= constraints.cos_sza_min

    return vis


def _compute_vis_batch(
    t_batch,
    keplerian_params: dict,
    propagator_type: str,
    gs_ecef: npt.NDArray,
    up: npt.NDArray,
    constraints: CoverageConstraints,
    use_fov: bool,
    pointing_lvlh_norm: npt.NDArray | None,
    cos_fov: float | None,
) -> npt.NDArray[np.bool_]:
    if use_fov:
        spec = SensorSpec(
            keplerian_params,
            propagator_type,
            "conic",
            pointing_lvlh_norm,
            cos_fov=cos_fov,
        )
    else:
        spec = SensorSpec(keplerian_params, propagator_type, "none", None)
    return _compute_vis_batch_multi(t_batch, [spec], gs_ecef, up, constraints)


def _make_offsets(
    t_start: np.datetime64,
    t_end: np.datetime64,
    max_step: np.timedelta64,
) -> tuple[npt.NDArray[np.int64], np.datetime64]:
    t_start = np.asarray(t_start, dtype="datetime64[us]")
    t_end = np.asarray(t_end, dtype="datetime64[us]")
    total_us = int((t_end - t_start) / np.timedelta64(1, "us"))
    step_us = int(max_step / np.timedelta64(1, "us"))
    if total_us <= 0 or step_us <= 0:
        return np.array([], dtype=np.int64), t_start
    offs = np.arange(0, total_us + 1, step_us, dtype=np.int64)
    if offs[-1] != total_us:
        offs = np.append(offs, np.int64(total_us))
    return offs, t_start


def _detect_transitions(
    vis_batch: npt.NDArray[np.bool_],
    in_access: npt.NDArray[np.bool_],
    t_batch: npt.NDArray,
) -> list[tuple[np.datetime64, int, bool]]:
    augmented = np.vstack([in_access[np.newaxis], vis_batch.astype(np.int8)]).astype(
        np.int8
    )
    diffs = np.diff(augmented, axis=0)
    rising_t, rising_m = np.where(diffs > 0)
    falling_t, falling_m = np.where(diffs < 0)
    all_t = np.concatenate([t_batch[rising_t], t_batch[falling_t]])
    all_m = np.concatenate([rising_m, falling_m])
    all_r = np.concatenate(
        [np.ones(len(rising_t), dtype=bool), np.zeros(len(falling_t), dtype=bool)]
    )
    order = np.argsort(all_t, kind="stable")
    return [(all_t[k], int(all_m[k]), bool(all_r[k])) for k in order]


def collect_access_intervals_multi(
    lat: npt.NDArray,
    lon: npt.NDArray,
    sensor_specs: list,
    t_start: np.datetime64,
    t_end: np.datetime64,
    alt: float,
    constraints: CoverageConstraints,
    max_step: np.timedelta64,
    batch_size: int,
    *,
    close_at_end: bool,
) -> list[list[tuple[np.datetime64, np.datetime64]]]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    M = len(lat)
    if M == 0:
        raise ValueError("lat/lon arrays must not be empty")

    gs_ecef, up = _build_gs(lat, lon, alt)

    offs, t_start_us = _make_offsets(t_start, t_end, max_step)
    N = len(offs)

    intervals: list[list[tuple[np.datetime64, np.datetime64]]] = [[] for _ in range(M)]
    current_aos: list[np.datetime64 | None] = [None] * M
    in_access = np.zeros(M, dtype=np.bool_)

    if N == 0:
        return intervals

    t_out = t_start_us + offs.astype("timedelta64[us]")

    for b0 in range(0, N, batch_size):
        b1 = min(b0 + batch_size, N)
        t_batch = t_out[b0:b1]
        vis = _compute_vis_batch_multi(
            t_batch,
            sensor_specs,
            gs_ecef,
            up,
            constraints,
        )

        for t_evt, m, is_rising in _detect_transitions(vis, in_access, t_batch):
            if is_rising:
                current_aos[m] = t_evt
            else:
                if current_aos[m] is not None:
                    intervals[m].append((current_aos[m], t_evt))
                    current_aos[m] = None

        in_access = vis[-1]

    if close_at_end:
        t_end_dt = np.datetime64(t_end, "us")
        for m in range(M):
            if in_access[m] and current_aos[m] is not None:
                intervals[m].append((current_aos[m], t_end_dt))

    return intervals
