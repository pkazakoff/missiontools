import numpy as np
import numpy.typing as npt

from .sampling import (
    sample_aoi,
    sample_region,
    sample_shapefile,
    sample_from_geometry,
    load_shapefile_geometry,
    sample_geography,
    geography_geometry,
)
from .visibility import (
    CoverageConstraints,
    SensorSpec,
    make_sensor_spec,
    make_sensor_spec_from_fov,
    _compute_vis_batch_multi,
    _build_gs,
    _make_offsets,
    collect_access_intervals_multi,
)


def coverage_fraction_multi(
    lat: npt.NDArray,
    lon: npt.NDArray,
    sensor_specs: list,
    t_start: np.datetime64,
    t_end: np.datetime64,
    alt: float,
    constraints: CoverageConstraints,
    max_step: np.timedelta64,
    batch_size: int,
) -> dict:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    M = len(lat)
    if M == 0:
        raise ValueError("lat/lon arrays must not be empty")

    gs_ecef, up = _build_gs(lat, lon, alt)

    offs, t_start_us = _make_offsets(t_start, t_end, max_step)
    N = len(offs)
    if N == 0:
        empty = np.array([], dtype=np.float32)
        return {
            "t": np.array([], dtype="datetime64[us]"),
            "fraction": empty,
            "cumulative": empty,
            "mean_fraction": float("nan"),
            "final_cumulative": float("nan"),
        }

    t_out = t_start_us + offs.astype("timedelta64[us]")
    frac_out = np.empty(N, dtype=np.float32)
    cum_out = np.empty(N, dtype=np.float32)

    ever_covered = np.zeros(M, dtype=np.bool_)
    n_covered = 0

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

        frac_out[b0:b1] = vis.mean(axis=1)

        for local_t in range(b1 - b0):
            new = vis[local_t] & ~ever_covered
            if new.any():
                ever_covered |= new
                n_covered += int(new.sum())
            cum_out[b0 + local_t] = n_covered / M

    return {
        "t": t_out,
        "fraction": frac_out,
        "cumulative": cum_out,
        "mean_fraction": float(np.mean(frac_out)),
        "final_cumulative": float(cum_out[-1]),
    }


def pointwise_coverage_multi(
    lat: npt.NDArray,
    lon: npt.NDArray,
    sensor_specs: list,
    t_start: np.datetime64,
    t_end: np.datetime64,
    alt: float,
    constraints: CoverageConstraints,
    max_step: np.timedelta64,
    batch_size: int,
) -> dict:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    M = len(lat)
    if M == 0:
        raise ValueError("lat/lon arrays must not be empty")

    gs_ecef, up = _build_gs(lat, lon, alt)

    offs, t_start_us = _make_offsets(t_start, t_end, max_step)
    N = len(offs)
    t_out = t_start_us + offs.astype("timedelta64[us]")

    if N == 0:
        return {
            "t": np.array([], dtype="datetime64[us]"),
            "lat": lat,
            "lon": lon,
            "alt": float(alt),
            "visible": np.empty((0, M), dtype=np.bool_),
        }

    visible = np.empty((N, M), dtype=np.bool_)

    for b0 in range(0, N, batch_size):
        b1 = min(b0 + batch_size, N)
        t_batch = t_out[b0:b1]
        visible[b0:b1] = _compute_vis_batch_multi(
            t_batch,
            sensor_specs,
            gs_ecef,
            up,
            constraints,
        )

    return {
        "t": t_out,
        "lat": lat,
        "lon": lon,
        "alt": float(alt),
        "visible": visible,
    }


def _collect_access_intervals(
    lat: npt.NDArray,
    lon: npt.NDArray,
    keplerian_params: dict,
    t_start: np.datetime64,
    t_end: np.datetime64,
    alt: float,
    constraints: CoverageConstraints,
    propagator_type: str,
    max_step: np.timedelta64,
    batch_size: int,
    fov_pointing_lvlh,
    fov_half_angle,
    *,
    close_at_end: bool,
) -> list[list[tuple[np.datetime64, np.datetime64]]]:
    spec = make_sensor_spec(
        keplerian_params, propagator_type, fov_pointing_lvlh, fov_half_angle
    )
    return collect_access_intervals_multi(
        lat,
        lon,
        [spec],
        t_start,
        t_end,
        alt,
        constraints,
        max_step,
        batch_size,
        close_at_end=close_at_end,
    )


def coverage_fraction(
    lat: npt.NDArray[np.floating],
    lon: npt.NDArray[np.floating],
    keplerian_params: dict,
    t_start: np.datetime64,
    t_end: np.datetime64,
    alt: float | np.floating = 0.0,
    el_min: float | np.floating = 0.0,
    propagator_type: str = "twobody",
    max_step: np.timedelta64 = np.timedelta64(30, "s"),
    batch_size: int = 1_000,
    fov_pointing_lvlh: npt.NDArray | None = None,
    fov_half_angle: float | None = None,
    sza_max: float | None = None,
    sza_min: float | None = None,
) -> dict:
    spec = make_sensor_spec(
        keplerian_params, propagator_type, fov_pointing_lvlh, fov_half_angle
    )
    constraints = CoverageConstraints.from_angles(el_min, sza_max, sza_min)
    return coverage_fraction_multi(
        lat,
        lon,
        [spec],
        t_start,
        t_end,
        alt,
        constraints,
        max_step,
        batch_size,
    )


def revisit_time(
    lat: npt.NDArray[np.floating],
    lon: npt.NDArray[np.floating],
    keplerian_params: dict,
    t_start: np.datetime64,
    t_end: np.datetime64,
    alt: float | np.floating = 0.0,
    el_min: float | np.floating = 0.0,
    propagator_type: str = "twobody",
    max_step: np.timedelta64 = np.timedelta64(30, "s"),
    batch_size: int = 1_000,
    fov_pointing_lvlh: npt.NDArray | None = None,
    fov_half_angle: float | None = None,
    sza_max: float | None = None,
    sza_min: float | None = None,
) -> dict:
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    M = len(lat)

    constraints = CoverageConstraints.from_angles(el_min, sza_max, sza_min)

    intervals = _collect_access_intervals(
        lat,
        lon,
        keplerian_params,
        t_start,
        t_end,
        alt,
        constraints,
        propagator_type,
        max_step,
        batch_size,
        fov_pointing_lvlh,
        fov_half_angle,
        close_at_end=False,
    )

    max_revisit = np.full(M, np.nan)
    mean_revisit = np.full(M, np.nan)
    for m, ivals in enumerate(intervals):
        if len(ivals) >= 2:
            gaps_us = np.array(
                [
                    int((ivals[i + 1][0] - ivals[i][1]) / np.timedelta64(1, "us"))
                    for i in range(len(ivals) - 1)
                ],
                dtype=np.float64,
            )
            max_revisit[m] = gaps_us.max() / 1e6
            mean_revisit[m] = gaps_us.mean() / 1e6

    has_gaps = ~np.isnan(max_revisit)
    return {
        "max_revisit": max_revisit,
        "mean_revisit": mean_revisit,
        "global_max": float(np.nanmax(max_revisit)) if has_gaps.any() else float("nan"),
        "global_mean": float(np.nanmean(mean_revisit))
        if has_gaps.any()
        else float("nan"),
    }


def pointwise_coverage(
    lat: npt.NDArray[np.floating],
    lon: npt.NDArray[np.floating],
    keplerian_params: dict,
    t_start: np.datetime64,
    t_end: np.datetime64,
    alt: float | np.floating = 0.0,
    el_min: float | np.floating = 0.0,
    propagator_type: str = "twobody",
    max_step: np.timedelta64 = np.timedelta64(30, "s"),
    batch_size: int = 1_000,
    fov_pointing_lvlh: npt.NDArray | None = None,
    fov_half_angle: float | None = None,
    sza_max: float | None = None,
    sza_min: float | None = None,
) -> dict:
    spec = make_sensor_spec(
        keplerian_params, propagator_type, fov_pointing_lvlh, fov_half_angle
    )
    constraints = CoverageConstraints.from_angles(el_min, sza_max, sza_min)
    return pointwise_coverage_multi(
        lat,
        lon,
        [spec],
        t_start,
        t_end,
        alt,
        constraints,
        max_step,
        batch_size,
    )


def access_pointwise(
    lat: npt.NDArray[np.floating],
    lon: npt.NDArray[np.floating],
    keplerian_params: dict,
    t_start: np.datetime64,
    t_end: np.datetime64,
    alt: float | np.floating = 0.0,
    el_min: float | np.floating = 0.0,
    propagator_type: str = "twobody",
    max_step: np.timedelta64 = np.timedelta64(30, "s"),
    batch_size: int = 1_000,
    fov_pointing_lvlh: npt.NDArray | None = None,
    fov_half_angle: float | None = None,
    sza_max: float | None = None,
    sza_min: float | None = None,
) -> list[list[tuple[np.datetime64, np.datetime64]]]:
    constraints = CoverageConstraints.from_angles(el_min, sza_max, sza_min)
    return _collect_access_intervals(
        lat,
        lon,
        keplerian_params,
        t_start,
        t_end,
        alt,
        constraints,
        propagator_type,
        max_step,
        batch_size,
        fov_pointing_lvlh,
        fov_half_angle,
        close_at_end=True,
    )


def revisit_pointwise(
    lat: npt.NDArray[np.floating],
    lon: npt.NDArray[np.floating],
    keplerian_params: dict,
    t_start: np.datetime64,
    t_end: np.datetime64,
    alt: float | np.floating = 0.0,
    el_min: float | np.floating = 0.0,
    propagator_type: str = "twobody",
    max_step: np.timedelta64 = np.timedelta64(30, "s"),
    batch_size: int = 1_000,
    fov_pointing_lvlh: npt.NDArray | None = None,
    fov_half_angle: float | None = None,
    sza_max: float | None = None,
    sza_min: float | None = None,
) -> list[npt.NDArray[np.timedelta64]]:
    constraints = CoverageConstraints.from_angles(el_min, sza_max, sza_min)
    intervals = _collect_access_intervals(
        lat,
        lon,
        keplerian_params,
        t_start,
        t_end,
        alt,
        constraints,
        propagator_type,
        max_step,
        batch_size,
        fov_pointing_lvlh,
        fov_half_angle,
        close_at_end=False,
    )
    result = []
    for ivals in intervals:
        if len(ivals) < 2:
            result.append(np.array([], dtype="timedelta64[us]"))
        else:
            gaps = np.array(
                [ivals[i + 1][0] - ivals[i][1] for i in range(len(ivals) - 1)],
                dtype="timedelta64[us]",
            )
            result.append(gaps)
    return result
