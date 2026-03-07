import numpy as np
import numpy.typing as npt
from typing import Callable

from .constants import EARTH_MEAN_RADIUS
from .frames import geodetic_to_ecef, eci_to_ecef
from .propagation import propagate_analytical
from ..cache import cached_propagate_analytical


def earth_access(vecs:   npt.NDArray[np.floating],
                 lat:    float | np.floating,
                 lon:    float | np.floating,
                 alt:    float | np.floating = 0.0,
                 el_min: float | np.floating = 0.0,
                 frame:  str = 'eci',
                 t:      npt.NDArray[np.datetime64] | None = None,
                 ) -> npt.NDArray[np.bool_]:
    """Determine which positions are visible from a ground station.

    A position is considered visible when the elevation angle from the ground
    station to that position is greater than or equal to ``el_min``.

    .. note::
        Earth blockage is not explicitly checked. For a surface station with
        ``el_min >= 0``, a positive elevation angle implies the line-of-sight
        clears the Earth. Sub-zero masks or elevated ground stations may
        require an additional ray–ellipsoid intersection check.

    Parameters
    ----------
    vecs : npt.NDArray[np.floating]
        Position vectors, shape ``(N, 3)`` (m).
    lat : float | np.floating
        Ground station geodetic latitude (rad).
    lon : float | np.floating
        Ground station longitude (rad).
    alt : float | np.floating, optional
        Ground station height above the WGS84 ellipsoid (m). Defaults to 0.
    el_min : float | np.floating, optional
        Minimum elevation angle (rad). Defaults to 0 (above the horizon).
    frame : str, optional
        Reference frame of ``vecs``: ``'eci'`` (default) or ``'ecef'``.
    t : npt.NDArray[np.datetime64] | None, optional
        UTC/UT1 observation times as ``datetime64[us]``, shape ``(N,)``.
        Required when ``frame='eci'``; ignored when ``frame='ecef'``.

    Returns
    -------
    npt.NDArray[np.bool_]
        Boolean array of shape ``(N,)``. ``True`` where the position is
        visible from the ground station above ``el_min``.

    Raises
    ------
    ValueError
        If ``frame='eci'`` and ``t`` is ``None``, or if ``frame`` is not
        ``'eci'`` or ``'ecef'``.
    """
    if frame == 'eci':
        if t is None:
            raise ValueError("t must be provided when frame='eci'")
        vecs_ecef = eci_to_ecef(vecs, t)
    elif frame == 'ecef':
        vecs_ecef = np.atleast_2d(vecs)
    else:
        raise ValueError(f"frame must be 'eci' or 'ecef', got '{frame!r}'")

    # Ground station ECEF position
    gs_ecef = geodetic_to_ecef(lat, lon, alt)  # (3,)

    # Outward ellipsoid normal at (lat, lon) — the geodetic "up" direction.
    # Using the geodetic normal (not gs_ecef / |gs_ecef|) gives elevation
    # angles consistent with a spirit level at the ground station.
    up = np.array([np.cos(lat) * np.cos(lon),
                   np.cos(lat) * np.sin(lon),
                   np.sin(lat)])  # unit vector

    # Line-of-sight from ground station to each position
    los = vecs_ecef - gs_ecef                             # (N, 3)
    los_unit = los / np.linalg.norm(los, axis=1, keepdims=True)

    # Elevation = arcsin of the component of los_unit along "up"
    sin_el = los_unit @ up                                # (N,)
    el = np.arcsin(np.clip(sin_el, -1.0, 1.0))

    return el >= el_min


# ---------------------------------------------------------------------------
# Internal shared interval-finding helper
# ---------------------------------------------------------------------------

def _find_intervals(
        access_fn: Callable[[npt.NDArray[np.int64]], npt.NDArray[np.bool_]],
        t_start: np.datetime64,
        t_end: np.datetime64,
        max_step: np.timedelta64,
        refine_tol: np.timedelta64,
        batch_size: int,
) -> list[tuple[np.datetime64, np.datetime64]]:
    """Find contiguous access intervals given a boolean access function.

    Parameters
    ----------
    access_fn : callable
        Accepts a 1-D array of µs-offsets from *t_start* and returns a
        boolean array of the same length indicating access at each offset.
    t_start, t_end : np.datetime64
        Search window.
    max_step : np.timedelta64
        Coarse-scan step size.
    refine_tol : np.timedelta64
        Binary-search convergence tolerance for edge times.
    batch_size : int
        Number of scan steps per propagation batch.

    Returns
    -------
    list[tuple[np.datetime64, np.datetime64]]
        ``(start, end)`` pairs for each continuous access window.
    """
    t_start = np.asarray(t_start, dtype='datetime64[us]')
    t_end   = np.asarray(t_end,   dtype='datetime64[us]')

    total_us = int((t_end - t_start) / np.timedelta64(1, 'us'))
    step_us  = int(max_step   / np.timedelta64(1, 'us'))
    tol_us   = int(refine_tol / np.timedelta64(1, 'us'))

    if total_us <= 0 or step_us <= 0:
        return []

    # Scan offsets (µs from t_start), always including t_end exactly
    offsets = np.arange(0, total_us + 1, step_us, dtype=np.int64)
    if offsets[-1] < total_us:
        offsets = np.append(offsets, np.int64(total_us))
    n_total = len(offsets)

    def t_at(off: int) -> np.datetime64:
        return t_start + np.timedelta64(int(off), 'us')

    def _refine_vectorized(
            transitions: list[tuple[int, int, bool]],
    ) -> list[int]:
        """Batch binary search: refine all transitions simultaneously."""
        if not transitions:
            return []
        tol = max(tol_us, 1)
        los    = np.array([t[0] for t in transitions], dtype=np.int64)
        his    = np.array([t[1] for t in transitions], dtype=np.int64)
        rising = np.array([t[2] for t in transitions], dtype=np.bool_)

        while np.any(his - los > tol):
            active      = his - los > tol
            mids        = los + (his - los) // 2
            active_idx  = np.where(active)[0]
            active_mids = mids[active_idx]
            flags       = access_fn(active_mids)
            match = rising[active_idx] == flags
            his[active_idx[match]]  = active_mids[match]
            los[active_idx[~match]] = active_mids[~match]

        return [int(his[j]) if rising[j] else int(los[j])
                for j in range(len(transitions))]

    # --- batched coarse scan ---
    intervals: list[tuple[np.datetime64, np.datetime64]] = []
    interval_start_us: int | None = None
    prev_flag: bool | None = None
    prev_offset: int = 0
    pending_transitions: list[tuple[int, int, bool]] = []

    for batch_start in range(0, n_total, batch_size):
        batch_end  = min(batch_start + batch_size, n_total)
        batch_offs = offsets[batch_start:batch_end]
        flags      = access_fn(batch_offs)

        if prev_flag is None:
            prev_flag   = bool(flags[0])
            prev_offset = int(batch_offs[0])
            if prev_flag:
                interval_start_us = prev_offset
            batch_offs = batch_offs[1:]
            flags      = flags[1:]

        if len(batch_offs) == 0:
            continue

        prev_and_flags = np.concatenate([[prev_flag], flags])
        change_k = np.where(prev_and_flags[:-1] != prev_and_flags[1:])[0]

        for k in change_k:
            lo     = prev_offset if k == 0 else int(batch_offs[k - 1])
            hi     = int(batch_offs[k])
            rising = not bool(prev_and_flags[k])
            pending_transitions.append((lo, hi, rising))

        prev_flag   = bool(flags[-1])
        prev_offset = int(batch_offs[-1])

    # --- vectorized refinement of all transitions at once ---
    refined = _refine_vectorized(pending_transitions)

    for idx, (_, _, rising) in enumerate(pending_transitions):
        if rising:
            interval_start_us = refined[idx]
        else:
            if interval_start_us is not None:
                intervals.append((t_at(interval_start_us), t_at(refined[idx])))
            interval_start_us = None

    # Close any interval still open at t_end
    if interval_start_us is not None:
        intervals.append((t_at(interval_start_us), t_end))

    return intervals


# ---------------------------------------------------------------------------
# Ground access
# ---------------------------------------------------------------------------

def earth_access_intervals(
        t_start:          np.datetime64,
        t_end:            np.datetime64,
        keplerian_params: dict,
        lat:              float | np.floating,
        lon:              float | np.floating,
        alt:              float | np.floating = 0.0,
        el_min:           float | np.floating = 0.0,
        propagator_type:  str = 'twobody',
        max_step:         np.timedelta64 = np.timedelta64(30, 's'),
        refine_tol:       np.timedelta64 = np.timedelta64(1, 's'),
        batch_size:       int = 10_000,
) -> list[tuple[np.datetime64, np.datetime64]]:
    """Find time intervals when a satellite is visible from a ground station.

    Performs a coarse scan at ``max_step`` cadence to detect access windows,
    then refines each rising/falling edge with binary search to within
    ``refine_tol``.

    .. warning::
        Passes shorter than ``max_step`` may be missed entirely. Set
        ``max_step`` to at most half the shortest expected pass duration.

    Parameters
    ----------
    t_start : np.datetime64
        Start of the search window (``datetime64[us]``).
    t_end : np.datetime64
        End of the search window (``datetime64[us]``).
    keplerian_params : dict
        Orbital elements at epoch. Must contain the keys ``epoch``, ``a``,
        ``e``, ``i``, ``arg_p``, ``raan``, ``ma``. Optionally
        ``central_body_mu``, ``central_body_j2``, ``central_body_radius``.
    lat : float | np.floating
        Ground station geodetic latitude (rad).
    lon : float | np.floating
        Ground station longitude (rad).
    alt : float | np.floating, optional
        Ground station height above the WGS84 ellipsoid (m). Defaults to 0.
    el_min : float | np.floating, optional
        Minimum elevation angle (rad). Defaults to 0 (horizon).
    propagator_type : str, optional
        ``'twobody'`` (default) or ``'j2'``.
    max_step : np.timedelta64, optional
        Maximum coarse scan step size. Defaults to 30 s.
    refine_tol : np.timedelta64, optional
        Binary-search convergence tolerance for edge times. Defaults to 1 s.
    batch_size : int, optional
        Number of scan steps per propagation batch. Limits peak memory usage
        to roughly ``batch_size × 24`` bytes. Defaults to 10 000.

    Returns
    -------
    list[tuple[np.datetime64, np.datetime64]]
        List of ``(start, end)`` pairs in ``datetime64[us]``, one per
        continuous access window. Empty list if no access occurs.
    """
    t_start = np.asarray(t_start, dtype='datetime64[us]')

    def _access_batch(off_arr: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
        t_arr = t_start + off_arr.astype('timedelta64[us]')
        r, _ = cached_propagate_analytical(t_arr, **keplerian_params,
                                           propagator_type=propagator_type)
        return earth_access(r, lat, lon, alt, el_min, frame='eci', t=t_arr)

    return _find_intervals(_access_batch, t_start, t_end,
                           max_step, refine_tol, batch_size)


# ---------------------------------------------------------------------------
# Space-to-space access
# ---------------------------------------------------------------------------

def _los_clear(r1_2d: npt.NDArray[np.floating],
               r2_2d: npt.NDArray[np.floating],
               body_radius: float,
               ) -> npt.NDArray[np.bool_]:
    """True where the line segment r1→r2 clears the spherical central body.

    Parameters
    ----------
    r1_2d, r2_2d : npt.NDArray[np.floating], shape ``(N, 3)``
        Position arrays with the central body centre at the origin.
    body_radius : float
        Radius of the obstructing sphere (m).

    Returns
    -------
    npt.NDArray[np.bool_], shape ``(N,)``
    """
    d    = r2_2d - r1_2d                              # (N, 3)
    d2   = np.einsum('ni,ni->n', d, d)                # |d|², (N,)

    # Guard against coincident points (d2 == 0): treat t* = 0, closest = r1.
    safe   = np.where(d2 > 0, d2, 1.0)
    t_star = np.clip(-np.einsum('ni,ni->n', r1_2d, d) / safe, 0.0, 1.0)

    closest = r1_2d + t_star[:, np.newaxis] * d       # (N, 3)
    dist2   = np.einsum('ni,ni->n', closest, closest)  # (N,)
    return dist2 >= body_radius ** 2


def space_to_space_access(
        r1:          npt.NDArray[np.floating],
        r2:          npt.NDArray[np.floating],
        body_radius: float = EARTH_MEAN_RADIUS,
) -> npt.NDArray[np.bool_]:
    """Determine which positions have an unobstructed line-of-sight.

    Checks whether the straight-line segment between each pair of positions
    clears the central body, modelled as a sphere of radius ``body_radius``.
    Both position arrays must be in the same reference frame with the central
    body centre at the origin (ECI and ECEF both satisfy this).

    .. note::
        A spherical obstruction model is used.  The maximum error relative to
        the WGS84 ellipsoid is ≤ 21 km (< 0.4 % of a typical orbit radius),
        which is negligible for access analysis.

        **Choosing** ``body_radius``:

        - ``EARTH_MEAN_RADIUS`` (default, 6 371 008 m) — best representative
          average; minimises neither false positives nor false negatives.
        - ``EARTH_SEMI_MAJOR_AXIS`` (6 378 137 m) — *conservative* for
          equatorial and mid-latitude paths.  A larger sphere is more likely
          to flag LOS as blocked near the Earth's limb, so this choice
          under-reports access (safe side for link planning).
        - A smaller radius such as ``EARTH_SEMI_MINOR_AXIS`` — conservative
          in the opposite direction; may slightly over-report access near the
          poles.

    Parameters
    ----------
    r1 : npt.NDArray[np.floating]
        First spacecraft position(s) (m), shape ``(N, 3)`` or ``(3,)``.
    r2 : npt.NDArray[np.floating]
        Second spacecraft position(s) (m), shape ``(N, 3)`` or ``(3,)``.
        Must be matched element-wise with ``r1``.
    body_radius : float, optional
        Radius of the obstructing central body sphere (m).
        Default: ``EARTH_MEAN_RADIUS`` (6 371 008 m).

    Returns
    -------
    npt.NDArray[np.bool_]
        Shape ``(N,)`` — ``True`` where the LOS is not blocked by the
        central body.  Returns a plain ``bool`` for scalar ``(3,)`` inputs.
    """
    r1a = np.asarray(r1, dtype=np.float64)
    r2a = np.asarray(r2, dtype=np.float64)
    scalar = r1a.ndim == 1
    result = _los_clear(np.atleast_2d(r1a), np.atleast_2d(r2a), body_radius)
    return bool(result[0]) if scalar else result


def space_to_space_access_intervals(
        t_start:             np.datetime64,
        t_end:               np.datetime64,
        keplerian_params_1:  dict,
        keplerian_params_2:  dict,
        body_radius:         float = EARTH_MEAN_RADIUS,
        propagator_type_1:   str = 'twobody',
        propagator_type_2:   str = 'twobody',
        max_step:            np.timedelta64 = np.timedelta64(30, 's'),
        refine_tol:          np.timedelta64 = np.timedelta64(1, 's'),
        batch_size:          int = 10_000,
) -> list[tuple[np.datetime64, np.datetime64]]:
    """Find time intervals when two spacecraft have unobstructed line-of-sight.

    Performs a coarse scan at ``max_step`` cadence to detect access windows,
    then refines each rising/falling edge with binary search to within
    ``refine_tol``.

    .. warning::
        Passes shorter than ``max_step`` may be missed entirely. Set
        ``max_step`` to at most half the shortest expected pass duration.

    Parameters
    ----------
    t_start : np.datetime64
        Start of the search window (``datetime64[us]``).
    t_end : np.datetime64
        End of the search window (``datetime64[us]``).
    keplerian_params_1 : dict
        Orbital elements of spacecraft 1. Same format as
        :func:`earth_access_intervals`.
    keplerian_params_2 : dict
        Orbital elements of spacecraft 2.
    body_radius : float, optional
        Radius of the obstructing central body sphere (m).
        Default: ``EARTH_MEAN_RADIUS``.
    propagator_type_1 : str, optional
        Propagator for spacecraft 1: ``'twobody'`` (default) or ``'j2'``.
    propagator_type_2 : str, optional
        Propagator for spacecraft 2: ``'twobody'`` (default) or ``'j2'``.
    max_step : np.timedelta64, optional
        Maximum coarse scan step size. Defaults to 30 s.
    refine_tol : np.timedelta64, optional
        Binary-search convergence tolerance for edge times. Defaults to 1 s.
    batch_size : int, optional
        Number of scan steps per propagation batch. Defaults to 10 000.

    Returns
    -------
    list[tuple[np.datetime64, np.datetime64]]
        List of ``(start, end)`` pairs in ``datetime64[us]``, one per
        continuous access window. Empty list if no access occurs.
    """
    t_start = np.asarray(t_start, dtype='datetime64[us]')

    def _access_batch(off_arr: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
        t_arr = t_start + off_arr.astype('timedelta64[us]')
        r1, _ = cached_propagate_analytical(t_arr, **keplerian_params_1,
                                            propagator_type=propagator_type_1)
        r2, _ = cached_propagate_analytical(t_arr, **keplerian_params_2,
                                            propagator_type=propagator_type_2)
        return space_to_space_access(r1, r2, body_radius)

    return _find_intervals(_access_batch, t_start, t_end,
                           max_step, refine_tol, batch_size)
