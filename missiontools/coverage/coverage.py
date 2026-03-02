import re as _re
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib.path import Path as _MplPath

from ..orbit.frames import geodetic_to_ecef, eci_to_ecef, lvlh_to_eci, sun_vec_eci
from ..orbit.propagation import propagate_analytical
from ..orbit.constants import EARTH_MEAN_RADIUS

# ---------------------------------------------------------------------------
# Bundled Natural Earth geodata paths
# ---------------------------------------------------------------------------

_GEODATA_DIR = Path(__file__).parent / 'geodata'
_NE_ADM0     = _GEODATA_DIR / 'ne_map_units'        / 'ne_50m_admin_0_map_units.shp'
_NE_ADM1     = _GEODATA_DIR / 'ne_states_provinces' / 'ne_50m_admin_1_states_provinces.shp'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fibonacci_sphere(n: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Equal-area Fibonacci lattice: n points on the unit sphere (radians)."""
    if n == 1:
        return np.array([0.0]), np.array([0.0])
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    i   = np.arange(n, dtype=np.float64)
    lat = np.arcsin(np.clip(2.0 * i / (n - 1) - 1.0, -1.0, 1.0))
    lon = (2.0 * np.pi * i / phi) % (2.0 * np.pi) - np.pi   # → (−π, π]
    return lat, lon


def _pip(polygon: npt.NDArray,
         lat:     npt.NDArray,
         lon:     npt.NDArray) -> npt.NDArray[np.bool_]:
    """Planar point-in-polygon test in lat/lon space (radians)."""
    # MplPath uses (x, y) = (lon, lat)
    path = _MplPath(polygon[:, ::-1])
    return path.contains_points(np.column_stack([lon, lat]))


def _build_gs(lat: npt.NDArray,
              lon: npt.NDArray,
              alt: float | npt.NDArray,
              ) -> tuple[npt.NDArray, npt.NDArray]:
    """Return ground-point ECEF positions (M,3) and geodetic up-vectors (M,3)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    gs_ecef = geodetic_to_ecef(lat, lon, alt)           # (M, 3)
    up = np.stack([np.cos(lat) * np.cos(lon),
                   np.cos(lat) * np.sin(lon),
                   np.sin(lat)], axis=-1)                # (M, 3)
    return gs_ecef, up


def _sun_ecef(t_arr: npt.NDArray) -> npt.NDArray:   # (T, 3)
    """Sun unit vector in ECEF at each timestep."""
    return eci_to_ecef(sun_vec_eci(t_arr), t_arr)


def _parse_constraints(
        fov_pointing_lvlh, fov_half_angle, sza_max, sza_min,
) -> tuple:
    """Validate FOV and SZA parameters; return precomputed values."""
    _fov_given = (fov_pointing_lvlh is not None, fov_half_angle is not None)
    if any(_fov_given) and not all(_fov_given):
        raise ValueError("fov_pointing_lvlh and fov_half_angle must both be "
                         "provided together")
    use_fov = all(_fov_given)
    pointing_lvlh_norm = cos_fov = None
    if use_fov:
        pointing_lvlh_norm = (np.asarray(fov_pointing_lvlh, dtype=np.float64)
                              / np.linalg.norm(fov_pointing_lvlh))
        cos_fov = float(np.cos(fov_half_angle))

    use_sza      = sza_max is not None or sza_min is not None
    cos_sza_max_ = float(np.cos(sza_max)) if sza_max is not None else None
    cos_sza_min_ = float(np.cos(sza_min)) if sza_min is not None else None

    return use_fov, pointing_lvlh_norm, cos_fov, use_sza, cos_sza_max_, cos_sza_min_


def _compute_vis_batch(
        t_batch,
        keplerian_params:  dict,
        propagator_type:   str,
        gs_ecef:           npt.NDArray,
        up:                npt.NDArray,
        sin_el_min:        float,
        use_fov:           bool,
        pointing_lvlh_norm: npt.NDArray | None,
        cos_fov:           float | None,
        use_sza:           bool,
        cos_sza_max_:      float | None,
        cos_sza_min_:      float | None,
) -> npt.NDArray[np.bool_]:              # (T, M)
    """Propagate one batch and return the visibility matrix."""
    r, v    = propagate_analytical(t_batch, **keplerian_params,
                                   type=propagator_type)
    pt_ecef = (_pointing_ecef(pointing_lvlh_norm, r, v, t_batch)
               if use_fov else None)
    sun_e   = _sun_ecef(t_batch) if use_sza else None
    return _visibility(r, t_batch, gs_ecef, up, sin_el_min,
                       pointing_ecef=pt_ecef,
                       cos_fov=cos_fov if use_fov else None,
                       sun_ecef=sun_e,
                       cos_sza_max=cos_sza_max_,
                       cos_sza_min=cos_sza_min_)


def _pointing_ecef(pointing_lvlh: npt.NDArray,   # (3,) unit vector in LVLH
                   r_eci:         npt.NDArray,   # (T, 3)
                   v_eci:         npt.NDArray,   # (T, 3)
                   t_arr:         npt.NDArray,   # (T,) datetime64[us]
                   ) -> npt.NDArray:             # (T, 3)
    """Convert a fixed LVLH pointing direction to ECEF at each timestep."""
    T     = len(t_arr)
    p_eci = lvlh_to_eci(np.tile(pointing_lvlh, (T, 1)), r_eci, v_eci)
    return eci_to_ecef(p_eci, t_arr)


def _visibility(r_eci:          npt.NDArray,         # (T, 3)
                t_arr:          npt.NDArray,         # (T,) datetime64[us]
                gs_ecef:        npt.NDArray,         # (M, 3)
                up:             npt.NDArray,         # (M, 3)
                sin_el_min:     float,
                pointing_ecef:  npt.NDArray | None = None,  # (T, 3)
                cos_fov:        float | None = None,
                sun_ecef:       npt.NDArray | None = None,  # (T, 3)
                cos_sza_max:    float | None = None,
                cos_sza_min:    float | None = None,
                ) -> npt.NDArray[np.bool_]:          # (T, M)
    """Vectorised visibility: T satellite positions × M ground points.

    Optional constraints (each ANDed with the elevation mask):

    * ``pointing_ecef`` / ``cos_fov`` — sensor FOV cone in ECEF
    * ``sun_ecef`` / ``cos_sza_max`` / ``cos_sza_min`` — solar zenith angle
    """
    r_ecef  = eci_to_ecef(r_eci, t_arr)                               # (T, 3)
    los     = r_ecef[:, np.newaxis, :] - gs_ecef[np.newaxis, :, :]   # (T, M, 3)
    norm    = np.maximum(np.linalg.norm(los, axis=2, keepdims=True), 1e-10)
    los_hat = los / norm                                               # (T, M, 3)
    sin_el  = np.einsum('tmi,mi->tm', los_hat, up)                    # (T, M)
    vis     = sin_el >= sin_el_min                                     # (T, M)

    if pointing_ecef is not None:
        # dot(sat→ground, pointing) = dot(-los_hat, pointing_ecef)
        fov_cos = np.einsum('tmi,ti->tm', -los_hat, pointing_ecef)    # (T, M)
        vis    &= fov_cos >= cos_fov

    if sun_ecef is not None:
        # cos(SZA) = dot(sun_ecef, up); cos decreasing so ≤ SZA ↔ ≥ cos
        cos_sza = np.einsum('ti,mi->tm', sun_ecef, up)                # (T, M)
        if cos_sza_max is not None:
            vis &= cos_sza >= cos_sza_max
        if cos_sza_min is not None:
            vis &= cos_sza <= cos_sza_min

    return vis


def _make_offsets(t_start:  np.datetime64,
                  t_end:    np.datetime64,
                  max_step: np.timedelta64,
                  ) -> tuple[npt.NDArray[np.int64], np.datetime64]:
    """Integer µs offsets from t_start, always including t_end."""
    t_start   = np.asarray(t_start, dtype='datetime64[us]')
    t_end     = np.asarray(t_end,   dtype='datetime64[us]')
    total_us  = int((t_end   - t_start) / np.timedelta64(1, 'us'))
    step_us   = int(max_step / np.timedelta64(1, 'us'))
    if total_us <= 0 or step_us <= 0:
        return np.array([], dtype=np.int64), t_start
    offs = np.arange(0, total_us + 1, step_us, dtype=np.int64)
    if offs[-1] != total_us:
        offs = np.append(offs, np.int64(total_us))
    return offs, t_start


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_aoi(
        polygon: npt.NDArray[np.floating],
        n:       int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Sample *n* approximately equal-area points inside an AoI polygon.

    Uses a Fibonacci lattice to generate a near-uniform global distribution,
    then filters to the points inside the polygon.  The returned count may
    differ slightly from *n* depending on AoI shape; use a larger *n* for
    denser or irregular regions.

    .. note::
        The polygon test is planar (lat/lon space in radians).  Results are
        accurate for regions that do not span the anti-meridian or enclose a
        pole.  Split such regions into separate polygons and combine samples.

    Parameters
    ----------
    polygon : npt.NDArray[np.floating]
        Polygon vertices, shape ``(V, 2)``, each row ``[lat, lon]`` in
        **radians**.
    n : int
        Target number of sample points.

    Returns
    -------
    lat : npt.NDArray[np.floating]
        Sample latitudes (rad), shape ``(M,)``.
    lon : npt.NDArray[np.floating]
        Sample longitudes (rad), shape ``(M,)``.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon must have shape (V, 2) with columns [lat, lon]")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")

    # Estimate AoI fraction from a pilot lattice
    n_pilot = max(n * 10, 5_000)
    lat_p, lon_p = _fibonacci_sphere(n_pilot)
    frac = float(_pip(polygon, lat_p, lon_p).mean())

    if frac < 1e-6:
        raise ValueError(
            "AoI polygon encloses too few global lattice points — "
            "check that coordinates are in radians and the polygon is not "
            "degenerate."
        )

    # Generate enough global points to get approximately n inside
    n_global = int(np.ceil(n / frac * 1.3))
    lat_all, lon_all = _fibonacci_sphere(n_global)
    inside  = _pip(polygon, lat_all, lon_all)
    lat_in  = lat_all[inside]
    lon_in  = lon_all[inside]

    # Evenly subsample if we ended up with more than n points
    if len(lat_in) > n:
        idx    = np.round(np.linspace(0, len(lat_in) - 1, n)).astype(int)
        lat_in = lat_in[idx]
        lon_in = lon_in[idx]

    return lat_in, lon_in


def sample_region(
        lat_min:       float | None = None,
        lat_max:       float | None = None,
        lon_min:       float | None = None,
        lon_max:       float | None = None,
        point_density: float = 1e11,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Sample an approximately equal-area Fibonacci lattice over a lat/lon band.

    A convenience wrapper around the Fibonacci lattice for rectangular
    regions.  The number of points is derived from the spherical zone area
    and the requested ``point_density``, so the grid automatically becomes
    denser for small regions and sparser for large ones.

    Unlike :func:`sample_aoi`, this function filters the lattice directly by
    coordinate bounds, so it handles anti-meridian-crossing regions and global
    or banded coverage correctly.

    Parameters
    ----------
    lat_min : float | None, optional
        Southern boundary (rad).  ``None`` extends to the South Pole (−π/2).
    lat_max : float | None, optional
        Northern boundary (rad).  ``None`` extends to the North Pole (+π/2).
    lon_min : float | None, optional
        Western boundary (rad).  Must be paired with ``lon_max``; ``None``
        (together with ``lon_max=None``) includes all longitudes.
    lon_max : float | None, optional
        Eastern boundary (rad).  Must be paired with ``lon_min``; ``None``
        (together with ``lon_min=None``) includes all longitudes.
        May be less than ``lon_min`` to indicate a region that crosses the
        anti-meridian (e.g. ``lon_min=np.radians(170)``,
        ``lon_max=np.radians(-170)``).
    point_density : float, optional
        Approximate area represented by each sample point (m²).
        Defaults to 1×10¹¹ m² (~100 000 km² per point, ~5 100 points
        globally).

    Returns
    -------
    lat : npt.NDArray[np.floating]
        Sample latitudes (rad), shape ``(M,)``.
    lon : npt.NDArray[np.floating]
        Sample longitudes (rad), shape ``(M,)``.

    Raises
    ------
    ValueError
        If exactly one of ``lon_min`` / ``lon_max`` is ``None``, if
        ``lat_min >= lat_max``, or if ``point_density`` is not positive.

    Examples
    --------
    Global coverage at ~100 000 km²/point::

        lat, lon = sample_region()

    Europe (approximate)::

        lat, lon = sample_region(np.radians(35), np.radians(70),
                                 np.radians(-10), np.radians(40),
                                 point_density=1e9)

    Pacific Ocean band crossing the anti-meridian::

        lat, lon = sample_region(np.radians(-30), np.radians(30),
                                 np.radians(150), np.radians(-120),
                                 point_density=1e10)
    """
    # --- validate longitude pairing ---
    if (lon_min is None) != (lon_max is None):
        raise ValueError(
            "lon_min and lon_max must both be None (all longitudes) or both "
            "be specified; got lon_min={} lon_max={}".format(lon_min, lon_max)
        )

    if point_density <= 0:
        raise ValueError(
            f"point_density must be positive, got {point_density}"
        )

    # --- resolve defaults ---
    lat_lo   = float(lat_min) if lat_min is not None else -np.pi / 2.0
    lat_hi   = float(lat_max) if lat_max is not None else  np.pi / 2.0
    full_lon = lon_min is None

    if lat_lo >= lat_hi:
        raise ValueError(
            f"lat_min ({lat_lo:.6f} rad) must be less than "
            f"lat_max ({lat_hi:.6f} rad)"
        )

    lon_lo = float(lon_min) if lon_min is not None else 0.0
    lon_hi = float(lon_max) if lon_max is not None else 0.0
    antimeridian = (not full_lon) and (lon_lo > lon_hi)

    # --- compute AoI area (spherical zone formula) ---
    # Area = 2π R² Δ(sin lat) × (lon span / 2π)
    if full_lon:
        lon_frac = 1.0
    elif antimeridian:
        lon_frac = (2.0 * np.pi - (lon_lo - lon_hi)) / (2.0 * np.pi)
    else:
        lon_frac = (lon_hi - lon_lo) / (2.0 * np.pi)

    area = (4.0 * np.pi * EARTH_MEAN_RADIUS**2
            * (np.sin(lat_hi) - np.sin(lat_lo)) / 2.0
            * lon_frac)

    n = max(1, int(np.round(area / point_density)))

    # --- generate a global Fibonacci lattice and filter ---
    # Oversample globally so that after filtering we have ≥ n points.
    area_fraction = area / (4.0 * np.pi * EARTH_MEAN_RADIUS**2)
    n_global = max(n * 5, int(np.ceil(n / area_fraction * 1.3)))
    lat_all, lon_all = _fibonacci_sphere(n_global)

    # Latitude filter
    lat_ok = (lat_all >= lat_lo) & (lat_all <= lat_hi)

    # Longitude filter
    if full_lon:
        lon_ok = np.ones(n_global, dtype=np.bool_)
    elif antimeridian:
        lon_ok = (lon_all >= lon_lo) | (lon_all <= lon_hi)
    else:
        lon_ok = (lon_all >= lon_lo) & (lon_all <= lon_hi)

    lat_in = lat_all[lat_ok & lon_ok]
    lon_in = lon_all[lat_ok & lon_ok]

    # Evenly subsample if we have more points than requested
    if len(lat_in) > n:
        idx    = np.round(np.linspace(0, len(lat_in) - 1, n)).astype(int)
        lat_in = lat_in[idx]
        lon_in = lon_in[idx]

    return lat_in, lon_in


def coverage_fraction(
        lat:               npt.NDArray[np.floating],
        lon:               npt.NDArray[np.floating],
        keplerian_params:  dict,
        t_start:           np.datetime64,
        t_end:             np.datetime64,
        alt:               float | np.floating = 0.0,
        el_min:            float | np.floating = 0.0,
        propagator_type:   str = 'twobody',
        max_step:          np.timedelta64 = np.timedelta64(30, 's'),
        batch_size:        int = 1_000,
        fov_pointing_lvlh: npt.NDArray | None = None,
        fov_half_angle:    float | None = None,
        sza_max:           float | None = None,
        sza_min:           float | None = None,
) -> dict:
    """Compute instantaneous and cumulative coverage fraction over a time window.

    For each sample epoch, the *instantaneous* fraction is the proportion of
    ground points with the satellite above ``el_min``.  The *cumulative*
    fraction is the proportion of ground points seen **at least once** up to
    that epoch.

    Parameters
    ----------
    lat, lon : npt.NDArray[np.floating]
        Ground-point latitudes/longitudes (rad), shape ``(M,)``.  Typically
        from :func:`sample_aoi`.
    keplerian_params : dict
        Orbital elements dict, e.g. from :func:`~missiontools.orbit.propagation.sun_synchronous_orbit`.
    t_start, t_end : np.datetime64
        Analysis window (``datetime64[us]``).
    alt : float | np.floating, optional
        Ground-point altitude above WGS84 (m).  Defaults to 0.
    el_min : float | np.floating, optional
        Minimum elevation angle (rad).  Defaults to 0 (horizon).
    propagator_type : str, optional
        ``'twobody'`` (default) or ``'j2'``.
    max_step : np.timedelta64, optional
        Scan time step.  Defaults to 30 s.
    batch_size : int, optional
        Time steps per propagation batch.  Defaults to 1 000.

    Returns
    -------
    dict
        ``t`` : ``(N,)`` ``datetime64[us]`` — sample timestamps.

        ``fraction`` : ``(N,)`` float — instantaneous coverage fraction.

        ``cumulative`` : ``(N,)`` float — cumulative coverage fraction.

        ``mean_fraction`` : float — time-averaged instantaneous fraction.

        ``final_cumulative`` : float — fraction of points covered ≥ once.
    """
    use_fov, pointing_lvlh_norm, cos_fov, use_sza, cos_sza_max_, cos_sza_min_ = \
        _parse_constraints(fov_pointing_lvlh, fov_half_angle, sza_max, sza_min)

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    M   = len(lat)
    if M == 0:
        raise ValueError("lat/lon arrays must not be empty")

    gs_ecef, up = _build_gs(lat, lon, alt)
    sin_el_min  = float(np.sin(el_min))

    offs, t_start_us = _make_offsets(t_start, t_end, max_step)
    N = len(offs)
    if N == 0:
        empty = np.array([], dtype=np.float32)
        return {
            't': np.array([], dtype='datetime64[us]'),
            'fraction': empty, 'cumulative': empty,
            'mean_fraction': float('nan'), 'final_cumulative': float('nan'),
        }

    t_out    = t_start_us + offs.astype('timedelta64[us]')   # (N,)
    frac_out = np.empty(N, dtype=np.float32)
    cum_out  = np.empty(N, dtype=np.float32)

    ever_covered = np.zeros(M, dtype=np.bool_)
    n_covered    = 0

    for b0 in range(0, N, batch_size):
        b1      = min(b0 + batch_size, N)
        t_batch = t_out[b0:b1]
        vis     = _compute_vis_batch(t_batch, keplerian_params, propagator_type,
                                     gs_ecef, up, sin_el_min,
                                     use_fov, pointing_lvlh_norm, cos_fov,
                                     use_sza, cos_sza_max_, cos_sza_min_)  # (T, M)

        frac_out[b0:b1] = vis.mean(axis=1)

        for local_t in range(b1 - b0):
            new = vis[local_t] & ~ever_covered
            if new.any():
                ever_covered |= new
                n_covered    += int(new.sum())
            cum_out[b0 + local_t] = n_covered / M

    return {
        't':                t_out,
        'fraction':         frac_out,
        'cumulative':       cum_out,
        'mean_fraction':    float(np.mean(frac_out)),
        'final_cumulative': float(cum_out[-1]),
    }


def revisit_time(
        lat:               npt.NDArray[np.floating],
        lon:               npt.NDArray[np.floating],
        keplerian_params:  dict,
        t_start:           np.datetime64,
        t_end:             np.datetime64,
        alt:               float | np.floating = 0.0,
        el_min:            float | np.floating = 0.0,
        propagator_type:   str = 'twobody',
        max_step:          np.timedelta64 = np.timedelta64(30, 's'),
        batch_size:        int = 1_000,
        fov_pointing_lvlh: npt.NDArray | None = None,
        fov_half_angle:    float | None = None,
        sza_max:           float | None = None,
        sza_min:           float | None = None,
) -> dict:
    """Compute per-point revisit time statistics over a time window.

    The *revisit time* for a ground point is the gap between loss of signal
    (LOS, last visible step) and the next acquisition of signal (AOS, first
    visible step on the following pass).  The initial gap from ``t_start`` to
    the first AOS is not included.

    .. note::
        Accuracy is limited to ``max_step``.  For exact edge times on
        individual points of interest use
        :func:`~missiontools.orbit.access.earth_access_intervals`.

    Parameters
    ----------
    lat, lon : npt.NDArray[np.floating]
        Ground-point latitudes/longitudes (rad), shape ``(M,)``.
    keplerian_params : dict
        Orbital elements dict.
    t_start, t_end : np.datetime64
        Analysis window.
    alt : float | np.floating, optional
        Ground-point altitude above WGS84 (m).  Defaults to 0.
    el_min : float | np.floating, optional
        Minimum elevation angle (rad).  Defaults to 0 (horizon).
    propagator_type : str, optional
        ``'twobody'`` or ``'j2'``.
    max_step : np.timedelta64, optional
        Scan step.  Defaults to 30 s.
    batch_size : int, optional
        Time steps per propagation batch.  Defaults to 1 000.

    Returns
    -------
    dict
        ``max_revisit`` : ``(M,)`` float — per-point max revisit time (s).
        ``nan`` for points accessed fewer than twice.

        ``mean_revisit`` : ``(M,)`` float — per-point mean revisit time (s).
        ``nan`` for points accessed fewer than twice.

        ``global_max`` : float — worst-case revisit time across all points (s).

        ``global_mean`` : float — mean of per-point mean revisit times (s).
    """
    use_fov, pointing_lvlh_norm, cos_fov, use_sza, cos_sza_max_, cos_sza_min_ = \
        _parse_constraints(fov_pointing_lvlh, fov_half_angle, sza_max, sza_min)

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    M   = len(lat)
    if M == 0:
        raise ValueError("lat/lon arrays must not be empty")

    gs_ecef, up = _build_gs(lat, lon, alt)
    sin_el_min  = float(np.sin(el_min))

    offs, t_start_us = _make_offsets(t_start, t_end, max_step)
    N = len(offs)
    _nan = np.full(M, np.nan)
    if N == 0:
        return {'max_revisit': _nan, 'mean_revisit': _nan,
                'global_max': float('nan'), 'global_mean': float('nan')}

    t_out = t_start_us + offs.astype('timedelta64[us]')  # (N,)

    # Per-point state (all µs offsets from t_start)
    in_access    = np.zeros(M, dtype=np.bool_)   # current access state
    had_los      = np.zeros(M, dtype=np.bool_)   # seen at least one LOS?
    prev_los_us  = np.zeros(M, dtype=np.int64)   # offset of most recent LOS
    max_gap_us   = np.zeros(M, dtype=np.int64)
    total_gap_us = np.zeros(M, dtype=np.int64)
    gap_count    = np.zeros(M, dtype=np.int64)

    for b0 in range(0, N, batch_size):
        b1      = min(b0 + batch_size, N)
        b_offs  = offs[b0:b1]                          # (T,) µs offsets
        t_batch = t_out[b0:b1]
        vis     = _compute_vis_batch(t_batch, keplerian_params, propagator_type,
                                     gs_ecef, up, sin_el_min,
                                     use_fov, pointing_lvlh_norm, cos_fov,
                                     use_sza, cos_sza_max_, cos_sza_min_)  # (T, M)

        # Detect transitions: prepend last-known state as row 0
        augmented = np.vstack([in_access[np.newaxis, :],
                               vis.astype(np.int8)]).astype(np.int8)  # (T+1, M)
        diffs = np.diff(augmented, axis=0)                             # (T,  M)
        rising_t,  rising_m  = np.where(diffs > 0)
        falling_t, falling_m = np.where(diffs < 0)

        # Merge into a time-ordered list of (offset_us, m, is_rising)
        all_offs = np.concatenate([b_offs[rising_t],  b_offs[falling_t]])
        all_m    = np.concatenate([rising_m,           falling_m])
        all_rise = np.concatenate([np.ones(len(rising_t),  dtype=np.bool_),
                                   np.zeros(len(falling_t), dtype=np.bool_)])
        order    = np.argsort(all_offs, kind='stable')

        for k in order:
            off_us   = int(all_offs[k])
            m        = int(all_m[k])
            is_rising = bool(all_rise[k])

            if is_rising:
                if had_los[m]:
                    gap = off_us - prev_los_us[m]
                    if gap > max_gap_us[m]:
                        max_gap_us[m] = gap
                    total_gap_us[m] += gap
                    gap_count[m]    += 1
            else:
                prev_los_us[m] = off_us
                had_los[m]     = True

        in_access[:] = vis[-1]

    # Convert µs → seconds; NaN for points with fewer than 2 accesses.
    # Guard gap_count==0 with np.where before dividing to avoid a spurious
    # RuntimeWarning (the guarded branch is discarded by the outer np.where).
    has_gaps = gap_count > 0
    safe_count = np.where(has_gaps, gap_count, 1)
    max_rev  = np.where(has_gaps, max_gap_us              / 1e6, np.nan)
    mean_rev = np.where(has_gaps, total_gap_us / safe_count / 1e6, np.nan)

    return {
        'max_revisit':  max_rev,
        'mean_revisit': mean_rev,
        'global_max':   float(np.nanmax(max_rev))  if has_gaps.any() else float('nan'),
        'global_mean':  float(np.nanmean(mean_rev)) if has_gaps.any() else float('nan'),
    }


def pointwise_coverage(
        lat:               npt.NDArray[np.floating],
        lon:               npt.NDArray[np.floating],
        keplerian_params:  dict,
        t_start:           np.datetime64,
        t_end:             np.datetime64,
        alt:               float | np.floating = 0.0,
        el_min:            float | np.floating = 0.0,
        propagator_type:   str = 'twobody',
        max_step:          np.timedelta64 = np.timedelta64(30, 's'),
        batch_size:        int = 1_000,
        fov_pointing_lvlh: npt.NDArray | None = None,
        fov_half_angle:    float | None = None,
        sza_max:           float | None = None,
        sza_min:           float | None = None,
) -> dict:
    """Return the raw (N × M) visibility matrix for every timestep and ground point.

    Unlike :func:`coverage_fraction` and :func:`revisit_time`, which reduce
    the visibility matrix to summary statistics, this function returns the
    full boolean matrix so callers can apply their own post-processing.

    Parameters
    ----------
    lat, lon : npt.NDArray[np.floating]
        Ground-point latitudes/longitudes (rad), shape ``(M,)``.
    keplerian_params : dict
        Orbital elements dict.
    t_start, t_end : np.datetime64
        Analysis window (``datetime64[us]``).
    alt : float | np.floating, optional
        Ground-point altitude above WGS84 (m).  Defaults to 0.
    el_min : float | np.floating, optional
        Minimum elevation angle (rad).  Defaults to 0 (horizon).
    propagator_type : str, optional
        ``'twobody'`` (default) or ``'j2'``.
    max_step : np.timedelta64, optional
        Scan time step.  Defaults to 30 s.
    batch_size : int, optional
        Time steps per propagation batch.  Defaults to 1 000.
    fov_pointing_lvlh : npt.NDArray | None, optional
        Sensor pointing direction in LVLH frame.  Must be paired with
        ``fov_half_angle``.
    fov_half_angle : float | None, optional
        FOV half-angle (rad).  Must be paired with ``fov_pointing_lvlh``.
    sza_max : float | None, optional
        Maximum solar zenith angle (rad).  Points where the SZA exceeds this
        value are considered invisible (daytime constraint).
    sza_min : float | None, optional
        Minimum solar zenith angle (rad).  Points where the SZA is below this
        value are considered invisible (nighttime constraint).

    Returns
    -------
    dict
        ``t`` : ``(N,)`` ``datetime64[us]`` — sample timestamps.

        ``lat`` : ``(M,)`` float — ground-point latitudes (rad), echoed from input.

        ``lon`` : ``(M,)`` float — ground-point longitudes (rad), echoed from input.

        ``alt`` : float — ground-point altitude (m), echoed from input.

        ``visible`` : ``(N, M)`` bool — ``True`` where the satellite has
        line-of-sight to the ground point at that timestep.

    Notes
    -----
    Memory usage scales as ``N × M`` booleans.  For a 30-day window at a
    20 s step (N ≈ 130 000) with M = 5 000 points this is roughly 650 MB.
    Prefer :func:`coverage_fraction` or :func:`revisit_time` for summary
    statistics over large grids.
    """
    use_fov, pointing_lvlh_norm, cos_fov, use_sza, cos_sza_max_, cos_sza_min_ = \
        _parse_constraints(fov_pointing_lvlh, fov_half_angle, sza_max, sza_min)

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    M   = len(lat)
    if M == 0:
        raise ValueError("lat/lon arrays must not be empty")

    gs_ecef, up = _build_gs(lat, lon, alt)
    sin_el_min  = float(np.sin(el_min))

    offs, t_start_us = _make_offsets(t_start, t_end, max_step)
    N = len(offs)
    t_out = t_start_us + offs.astype('timedelta64[us]')   # (N,)

    if N == 0:
        return {
            't':       np.array([], dtype='datetime64[us]'),
            'lat':     lat,
            'lon':     lon,
            'alt':     float(alt),
            'visible': np.empty((0, M), dtype=np.bool_),
        }

    visible = np.empty((N, M), dtype=np.bool_)

    for b0 in range(0, N, batch_size):
        b1      = min(b0 + batch_size, N)
        t_batch = t_out[b0:b1]
        visible[b0:b1] = _compute_vis_batch(
            t_batch, keplerian_params, propagator_type,
            gs_ecef, up, sin_el_min,
            use_fov, pointing_lvlh_norm, cos_fov,
            use_sza, cos_sza_max_, cos_sza_min_,
        )

    return {
        't':       t_out,
        'lat':     lat,
        'lon':     lon,
        'alt':     float(alt),
        'visible': visible,
    }


# ---------------------------------------------------------------------------
# Shapefile helpers
# ---------------------------------------------------------------------------

#: ESRI shapetype codes for polygon variants (Polygon, PolygonZ, PolygonM)
_SHP_POLYGON_TYPES = frozenset({5, 15, 25})


def _load_shapefile(path, feature_index):
    """Read an ESRI Shapefile and return a shapely geometry.

    Returns
    -------
    geom : shapely geometry
        Union of all selected features (Polygon or MultiPolygon).
        Rings are *unwrapped* so antimeridian-crossing polygons have
        continuous (possibly out-of-[-180,180]) coordinates.
    crosses_am : bool
        True if any ring required longitude unwrapping, indicating that
        the geometry may extend outside the [-180, 180] longitude range
        and requires shifted-point PIP tests.
    """
    import shapefile as _pyshp
    from shapely.geometry import Polygon as _Polygon
    from shapely.ops import unary_union as _unary_union

    sf = _pyshp.Reader(str(path))

    if feature_index is not None:
        raw_shapes = [sf.shape(feature_index)]
    else:
        raw_shapes = sf.shapes()

    crosses_am = False
    geoms = []

    for shp in raw_shapes:
        if shp.shapeType not in _SHP_POLYGON_TYPES:
            continue

        # Split the flat point list into per-part rings using the parts offsets
        part_starts = list(shp.parts) + [len(shp.points)]
        rings_raw = [
            shp.points[part_starts[k]: part_starts[k + 1]]
            for k in range(len(part_starts) - 1)
        ]

        # Unwrap each ring: remove antimeridian longitude jumps so the ring
        # is represented as a continuous (but possibly extended) coordinate set.
        unwrapped_rings = []
        for ring in rings_raw:
            lons_raw = [pt[0] for pt in ring]
            lats     = [pt[1] for pt in ring]
            lons_u   = [lons_raw[0]]
            for j in range(1, len(lons_raw)):
                dl = lons_raw[j] - lons_u[-1]
                if dl > 180.0:
                    lons_u.append(lons_raw[j] - 360.0)
                    crosses_am = True
                elif dl < -180.0:
                    lons_u.append(lons_raw[j] + 360.0)
                    crosses_am = True
                else:
                    lons_u.append(lons_raw[j])
            unwrapped_rings.append(list(zip(lons_u, lats)))

        if unwrapped_rings:
            exterior = unwrapped_rings[0]
            holes    = unwrapped_rings[1:]
            geoms.append(_Polygon(exterior, holes))

    if not geoms:
        raise ValueError(
            "No polygon features found in the shapefile.  "
            "Only shapeTypes 5 (Polygon), 15 (PolygonZ), and 25 (PolygonM) "
            "are supported."
        )

    geom = _unary_union(geoms) if len(geoms) > 1 else geoms[0]
    return geom, crosses_am


def _sample_from_geom(
        geom,
        crosses_am: bool,
        point_density: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Fibonacci-sphere sampling inside a shapely geometry.

    Parameters
    ----------
    geom : shapely Polygon or MultiPolygon
        Target geometry in geographic degrees (may have extended longitudes
        if the polygon crosses the antimeridian).
    crosses_am : bool
        If True, PIP tests are also performed at ±360° longitude shifts.
    point_density : float
        Approximate area per sample point (m²).  Must be positive.

    Returns
    -------
    lat : (M,) float64, radians
    lon : (M,) float64, radians
    """
    import shapely as _shapely

    # Area estimate from bounding box (spherical zone formula)
    minx, miny, maxx, maxy = geom.bounds   # degrees
    lat_lo = max(np.radians(miny), -np.pi / 2.0)
    lat_hi = min(np.radians(maxy),  np.pi / 2.0)
    lon_span_rad = np.radians(min(abs(maxx - minx), 360.0))
    area_approx = (4.0 * np.pi * EARTH_MEAN_RADIUS**2
                   * (np.sin(lat_hi) - np.sin(lat_lo)) / 2.0
                   * lon_span_rad / (2.0 * np.pi))

    n = max(1, int(np.round(area_approx / point_density)))

    # Oversample globally and filter via PIP
    area_frac  = max(area_approx / (4.0 * np.pi * EARTH_MEAN_RADIUS**2), 1e-9)
    n_global   = max(n * 5, int(np.ceil(n / area_frac * 1.3)))
    lat_r, lon_r = _fibonacci_sphere(n_global)

    # Shapely expects degrees (lon, lat) = (x, y)
    lon_deg = np.degrees(lon_r)
    lat_deg = np.degrees(lat_r)

    inside = _shapely.contains_xy(geom, lon_deg, lat_deg)
    if crosses_am:
        # Also test points shifted by ±360° to reach the unwrapped polygon
        inside |= _shapely.contains_xy(geom, lon_deg + 360.0, lat_deg)
        inside |= _shapely.contains_xy(geom, lon_deg - 360.0, lat_deg)

    lat_in = lat_r[inside]
    lon_in = lon_r[inside]

    if len(lat_in) == 0:
        raise ValueError(
            "No sample points fell inside the geometry — "
            "check that coordinates are in geographic degrees (WGS84 / EPSG:4326)."
        )

    if len(lat_in) > n:
        idx    = np.round(np.linspace(0, len(lat_in) - 1, n)).astype(int)
        lat_in = lat_in[idx]
        lon_in = lon_in[idx]

    return lat_in, lon_in


def sample_shapefile(
        path: str,
        *,
        feature_index: int | None = None,
        point_density: float = 1e11,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Sample approximately equal-area points from an ESRI Shapefile polygon.

    Reads the polygon geometry from a ``.shp`` file and returns a Fibonacci
    lattice filtered to the interior, using the same density convention as
    :func:`sample_region`.  Antimeridian-crossing polygons (e.g. Pacific
    island groups, Russia) are handled correctly via coordinate unwrapping
    and shifted-point PIP tests.

    Parameters
    ----------
    path : str
        Path to the ``.shp`` file.
    feature_index : int or None, optional
        Index of the feature/shape to sample from.  ``None`` (default)
        unions all features in the layer.
    point_density : float, optional
        Approximate area represented by each sample point (m²).
        Defaults to 1×10¹¹ m² (~100 000 km² per point).

    Returns
    -------
    lat : npt.NDArray[np.floating]
        Sample latitudes (rad), shape ``(M,)``.
    lon : npt.NDArray[np.floating]
        Sample longitudes (rad), shape ``(M,)``.

    Raises
    ------
    ValueError
        If the file contains no polygon features, or if no lattice points
        land inside the geometry (degenerate or very small region).

    Notes
    -----
    Requires ``pyshp`` and ``shapely`` (both declared as package
    dependencies).
    """
    if point_density <= 0:
        raise ValueError(f"point_density must be positive, got {point_density}")

    geom, crosses_am = _load_shapefile(path, feature_index)
    return _sample_from_geom(geom, crosses_am, point_density)


# ---------------------------------------------------------------------------
# Natural Earth geography helpers
# ---------------------------------------------------------------------------

def _load_ne_features(
        path: str,
        indices: list[int],
) -> tuple:
    """Load and union a specific subset of features from a Natural Earth shapefile.

    Parameters
    ----------
    path : str
        Path to the ``.shp`` file.
    indices : list[int]
        Feature indices to load and union.

    Returns
    -------
    geom : shapely Polygon or MultiPolygon
    crosses_am : bool
    """
    from shapely.ops import unary_union as _unary_union

    geoms: list = []
    crosses_am = False
    for i in indices:
        g, cam = _load_shapefile(path, i)
        geoms.append(g)
        crosses_am = crosses_am or cam

    if not geoms:
        raise ValueError("No features matched the requested indices.")

    return (_unary_union(geoms) if len(geoms) > 1 else geoms[0]), crosses_am


def _find_ne_indices(geography: str) -> tuple[str, list[int]]:
    """Resolve a geography string to a shapefile path and feature indices.

    Detection order
    ---------------
    1. Slash pattern  ``'Country/Subdivision'``  → admin-1 by name
    2. ISO 3166-2     ``'CA-QC'``               → admin-1 ``iso_3166_2``
    3. ISO A2         ``'CA'``                   → admin-0 ``ISO_A2``
    4. ISO A3         ``'CAN'``                  → admin-0 ``ISO_A3``
    5. Name           ``'Canada'``               → admin-0 ``NAME`` (case-insensitive);
                                                   fallback to admin-1 ``name``

    Returns
    -------
    path : str
        Absolute path to the matched shapefile.
    indices : list[int]
        Feature indices (one or more) to union.

    Raises
    ------
    ValueError
        If the geography string matches none of the above.
    """
    import shapefile as _pyshp

    g = geography.strip()

    # ── 1. Slash: "Country/Subdivision" ─────────────────────────────────────
    if '/' in g:
        country, subdivision = (s.strip() for s in g.split('/', 1))
        cl = country.lower()
        sl = subdivision.lower()
        sf1 = _pyshp.Reader(str(_NE_ADM1))
        idx = [i for i, r in enumerate(sf1.records())
               if r.as_dict()['admin'].lower() == cl
               and (r.as_dict()['name'].lower()    == sl
                    or r.as_dict()['name_en'].lower() == sl)]
        if not idx:
            raise ValueError(
                f"Subdivision {subdivision!r} not found in {country!r}. "
                f"Sub-national (admin-1) data is available only for: "
                f"Australia, Brazil, Canada, China, India, Indonesia, "
                f"Russia, South Africa, United States of America."
            )
        return str(_NE_ADM1), idx

    # ── 2. ISO 3166-2: "CA-QC" ──────────────────────────────────────────────
    if _re.match(r'^[A-Z]{2}-[A-Z0-9]{1,3}$', g):
        sf1 = _pyshp.Reader(str(_NE_ADM1))
        idx = [i for i, r in enumerate(sf1.records())
               if r.as_dict()['iso_3166_2'] == g]
        if idx:
            return str(_NE_ADM1), idx

    # ── 3. ISO A2: "CA" ─────────────────────────────────────────────────────
    if len(g) == 2 and g.isupper():
        sf0 = _pyshp.Reader(str(_NE_ADM0))
        idx = [i for i, r in enumerate(sf0.records())
               if r.as_dict()['ISO_A2'] == g]
        if idx:
            return str(_NE_ADM0), idx

    # ── 4. ISO A3: "CAN" ────────────────────────────────────────────────────
    if len(g) == 3 and g.isupper():
        sf0 = _pyshp.Reader(str(_NE_ADM0))
        idx = [i for i, r in enumerate(sf0.records())
               if r.as_dict()['ISO_A3'] == g]
        if idx:
            return str(_NE_ADM0), idx

    # ── 5. Name lookup (case-insensitive) ────────────────────────────────────
    gl = g.lower()

    sf0 = _pyshp.Reader(str(_NE_ADM0))
    idx = [i for i, r in enumerate(sf0.records())
           if r.as_dict()['NAME'].lower() == gl]
    if idx:
        return str(_NE_ADM0), idx

    sf1 = _pyshp.Reader(str(_NE_ADM1))
    idx = [i for i, r in enumerate(sf1.records())
           if r.as_dict()['name'].lower()    == gl
           or r.as_dict()['name_en'].lower() == gl]
    if idx:
        return str(_NE_ADM1), idx

    raise ValueError(
        f"Geography not found: {geography!r}. "
        f"Accepted formats: country name ('Canada'), "
        f"'Country/Subdivision' ('Canada/Quebec'), "
        f"ISO A2 ('CA'), ISO A3 ('CAN'), ISO 3166-2 ('CA-QC')."
    )


def sample_geography(
        geography: str,
        *,
        point_density: float = 1e11,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], object]:
    """Sample approximately equal-area points from a Natural Earth geography.

    Looks up the requested country or subdivision in the bundled Natural Earth
    50 m dataset and returns a Fibonacci lattice filtered to its interior, using
    the same density convention as :func:`sample_region`.

    Parameters
    ----------
    geography : str
        The geography to sample.  Accepted formats (auto-detected):

        - Country name: ``'Canada'`` (case-insensitive, matched against the
          admin-0 ``NAME`` field; falls back to admin-1 ``name`` if no match).
        - ``'Country/Subdivision'``: ``'Canada/Quebec'``, ``'United States of
          America/Alaska'``.  Sub-national data is available only for
          Australia, Brazil, Canada, China, India, Indonesia, Russia,
          South Africa, and the United States of America.
        - ISO 3166-1 alpha-2: ``'CA'``, ``'US'``
        - ISO 3166-1 alpha-3: ``'CAN'``, ``'USA'``
        - ISO 3166-2: ``'CA-QC'``, ``'US-AK'``
    point_density : float, optional
        Approximate area represented by each sample point (m²).
        Defaults to 1×10¹¹ m² (~100 000 km² per point).

    Returns
    -------
    lat : (M,) float64
        Sample latitudes (rad).
    lon : (M,) float64
        Sample longitudes (rad).
    geometry : shapely Polygon or MultiPolygon
        Shapely geometry of the matched feature(s).

    Raises
    ------
    ValueError
        If ``geography`` does not match any feature, ``point_density`` is
        non-positive, or no lattice points fall inside the geometry.

    Examples
    --------
    ::

        lat, lon, geom = sample_geography('Canada')
        lat, lon, geom = sample_geography('Canada/Quebec')
        lat, lon, geom = sample_geography('CA-QC')
        lat, lon, geom = sample_geography('CAN')
    """
    if point_density <= 0:
        raise ValueError(f"point_density must be positive, got {point_density}")

    path, indices = _find_ne_indices(geography)
    geom, crosses_am = _load_ne_features(path, indices)
    lat, lon = _sample_from_geom(geom, crosses_am, point_density)
    return lat, lon, geom
