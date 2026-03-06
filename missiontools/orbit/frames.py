import numpy as np
import numpy.typing as npt

from .constants import EARTH_SEMI_MAJOR_AXIS, EARTH_INVERSE_FLATTENING

# J2000.0 epoch in UTC (≈ UT1 to < 1 s; TT leads UT1 by ~69 s and should NOT be used)
_J2000_US = np.datetime64('2000-01-01T12:00:00', 'us')
_SECONDS_PER_JULIAN_CENTURY = 36525.0 * 86400.0


def gmst(t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.floating]:
    """Greenwich Mean Sidereal Time (rad) from an array of UTC/UT1 datetimes.

    Uses the IAU 1982 polynomial. T is computed directly from the integer
    microsecond offset from J2000.0, avoiding the ~40 µs precision floor of
    a single-precision Julian Date float.

    Parameters
    ----------
    t : npt.NDArray[np.datetime64]
        Observation times as ``datetime64[us]``. Values are interpreted as
        **UT1** (passing UTC introduces < 0.004° error; passing TT introduces
        ~0.29° error and should be avoided).

    Returns
    -------
    npt.NDArray[np.floating]
        GMST in radians, wrapped to [0, 2π).
    """
    t_us = np.asarray(t, dtype='datetime64[us]')

    # Seconds from J2000.0 — int64 microsecond difference cast to float64
    # preserves ~0.1 µs precision (vs ~40 µs for a single JD float64)
    s = (t_us - _J2000_US).astype(np.float64) * 1e-6

    # Julian centuries from J2000.0
    T = s / _SECONDS_PER_JULIAN_CENTURY

    # IAU 1982 GMST polynomial — result in seconds of time
    theta = (67310.54841
             + (876600 * 3600 + 8640184.812866) * T
             + 0.093104 * T**2
             - 6.2e-6   * T**3)

    # Seconds of time → radians (1 s = 1/240 °), wrapped to [0, 2π)
    return (np.deg2rad(theta / 240.0)) % (2 * np.pi)

def eci_to_ecef(eci_vecs: npt.NDArray[np.floating],
                t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.floating]:
    """Convert ECI position/velocity vectors to ECEF via GMST rotation.

    Parameters
    ----------
    eci_vecs : npt.NDArray[np.floating]
        Vectors in the ECI frame, shape ``(N, 3)`` or ``(3,)`` for a single
        vector.
    t : npt.NDArray[np.datetime64]
        UTC/UT1 observation times as ``datetime64[us]``, shape ``(N,)`` or
        scalar. Must match the first dimension of ``eci_vecs``.

    Returns
    -------
    npt.NDArray[np.floating]
        Vectors in the ECEF frame, same shape as ``eci_vecs``.
    """
    theta = gmst(t)
    scalar = np.ndim(theta) == 0
    theta = np.atleast_1d(theta)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    z, o = np.zeros_like(theta), np.ones_like(theta)

    # Rz(-θ): ECI → ECEF rotates the frame eastward by the GMST angle
    Rz = np.array([[ cos_t,  sin_t, z],
                   [-sin_t,  cos_t, z],
                   [     z,      z, o]]).transpose(2, 0, 1)  # (N, 3, 3)

    result = np.einsum('nij,nj->ni', Rz, np.atleast_2d(eci_vecs))  # (N, 3)
    return result[0] if scalar else result


def ecef_to_eci(ecef_vecs: npt.NDArray[np.floating],
                t: npt.NDArray[np.datetime64]) -> npt.NDArray[np.floating]:
    """Convert ECEF position/velocity vectors to ECI via GMST rotation.

    Parameters
    ----------
    ecef_vecs : npt.NDArray[np.floating]
        Vectors in the ECEF frame, shape ``(N, 3)`` or ``(3,)`` for a single
        vector.
    t : npt.NDArray[np.datetime64]
        UTC/UT1 observation times as ``datetime64[us]``, shape ``(N,)`` or
        scalar. Must match the first dimension of ``ecef_vecs``.

    Returns
    -------
    npt.NDArray[np.floating]
        Vectors in the ECI frame, same shape as ``ecef_vecs``.
    """
    theta = gmst(t)
    scalar = np.ndim(theta) == 0
    theta = np.atleast_1d(theta)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    z, o = np.zeros_like(theta), np.ones_like(theta)

    # Rz(+θ) = Rz(-θ)ᵀ: ECEF → ECI is the transpose of the ECI→ECEF rotation
    Rz = np.array([[cos_t, -sin_t, z],
                   [sin_t,  cos_t, z],
                   [    z,      z, o]]).transpose(2, 0, 1)  # (N, 3, 3)

    result = np.einsum('nij,nj->ni', Rz, np.atleast_2d(ecef_vecs))  # (N, 3)
    return result[0] if scalar else result


def geodetic_to_ecef(lat: float | npt.NDArray[np.floating],
                     lon: float | npt.NDArray[np.floating],
                     alt: float | npt.NDArray[np.floating] = 0.0
                     ) -> npt.NDArray[np.floating]:
    """Convert geodetic coordinates to ECEF Cartesian coordinates (WGS84).

    Parameters
    ----------
    lat : float | npt.NDArray[np.floating]
        Geodetic latitude (rad), scalar or shape ``(N,)``.
    lon : float | npt.NDArray[np.floating]
        Longitude (rad), scalar or shape ``(N,)``.
    alt : float | npt.NDArray[np.floating], optional
        Height above the WGS84 ellipsoid (m). Defaults to 0.

    Returns
    -------
    npt.NDArray[np.floating]
        ECEF position vector(s) (m), shape ``(3,)`` for scalar inputs or
        ``(N, 3)`` for array inputs.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)
    scalar = lat.ndim == 0 and lon.ndim == 0 and alt.ndim == 0

    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)
    alt = np.atleast_1d(alt)

    # WGS84 derived constants
    f  = 1.0 / EARTH_INVERSE_FLATTENING
    e2 = 2.0 * f - f**2          # first eccentricity squared
    a  = EARTH_SEMI_MAJOR_AXIS

    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)

    x = (N + alt)            * np.cos(lat) * np.cos(lon)
    y = (N + alt)            * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - e2) + alt) * np.sin(lat)

    result = np.stack([x, y, z], axis=-1)  # (N, 3)
    return result[0] if scalar else result


def _lvlh_basis(r_eci: npt.NDArray[np.floating],
                v_eci: npt.NDArray[np.floating],
                ) -> npt.NDArray[np.floating]:
    """Build the (N, 3, 3) ECI→LVLH rotation matrix Q for each state vector.

    Rows of Q are the three LVLH unit vectors expressed in ECI coordinates:
    row 0 = R̂ (radial), row 1 = Ŝ (along-track), row 2 = Ŵ (orbit normal).
    """
    R_hat = r_eci / np.linalg.norm(r_eci, axis=1, keepdims=True)    # (N, 3)
    h     = np.cross(r_eci, v_eci)                                   # (N, 3)
    W_hat = h     / np.linalg.norm(h,     axis=1, keepdims=True)    # (N, 3)
    S_hat = np.cross(W_hat, R_hat)                                   # (N, 3)
    return np.stack([R_hat, S_hat, W_hat], axis=1)                   # (N, 3, 3)


def eci_to_lvlh(vecs:  npt.NDArray[np.floating],
                r_eci: npt.NDArray[np.floating],
                v_eci: npt.NDArray[np.floating],
                ) -> npt.NDArray[np.floating]:
    """Convert vectors from the ECI frame to the LVLH (RSW) frame.

    The LVLH frame is defined by the satellite's instantaneous orbital state:

    * **x̂ (R)** — radial, pointing away from the central body
    * **ŷ (S)** — along-track, in the orbital plane (= velocity direction
      for circular orbits)
    * **ẑ (W)** — cross-track/normal, in the angular-momentum direction
      (right-hand normal to the orbital plane)

    Parameters
    ----------
    vecs : npt.NDArray[np.floating]
        Vectors to transform in the ECI frame, shape ``(N, 3)`` or ``(3,)``.
    r_eci : npt.NDArray[np.floating]
        Satellite ECI position vector(s), shape ``(N, 3)`` or ``(3,)`` (m).
    v_eci : npt.NDArray[np.floating]
        Satellite ECI velocity vector(s), shape ``(N, 3)`` or ``(3,)`` (m/s).

    Returns
    -------
    npt.NDArray[np.floating]
        Vectors in the LVLH frame, same shape as ``vecs``.
    """
    vecs  = np.asarray(vecs,  dtype=np.float64)
    r_eci = np.asarray(r_eci, dtype=np.float64)
    v_eci = np.asarray(v_eci, dtype=np.float64)
    scalar = vecs.ndim == 1
    Q      = _lvlh_basis(np.atleast_2d(r_eci), np.atleast_2d(v_eci))  # (N,3,3)
    result = np.einsum('nij,nj->ni', Q, np.atleast_2d(vecs))           # (N, 3)
    return result[0] if scalar else result


def lvlh_to_eci(vecs:  npt.NDArray[np.floating],
                r_eci: npt.NDArray[np.floating],
                v_eci: npt.NDArray[np.floating],
                ) -> npt.NDArray[np.floating]:
    """Convert vectors from the LVLH (RSW) frame to the ECI frame.

    Inverse of :func:`eci_to_lvlh`.  Because the LVLH rotation matrix Q is
    orthonormal, the inverse is simply its transpose: ``v_eci = Qᵀ v_lvlh``.

    Parameters
    ----------
    vecs : npt.NDArray[np.floating]
        Vectors to transform in the LVLH frame, shape ``(N, 3)`` or ``(3,)``.
    r_eci : npt.NDArray[np.floating]
        Satellite ECI position vector(s), shape ``(N, 3)`` or ``(3,)`` (m).
    v_eci : npt.NDArray[np.floating]
        Satellite ECI velocity vector(s), shape ``(N, 3)`` or ``(3,)`` (m/s).

    Returns
    -------
    npt.NDArray[np.floating]
        Vectors in the ECI frame, same shape as ``vecs``.
    """
    vecs  = np.asarray(vecs,  dtype=np.float64)
    r_eci = np.asarray(r_eci, dtype=np.float64)
    v_eci = np.asarray(v_eci, dtype=np.float64)
    scalar = vecs.ndim == 1
    Q      = _lvlh_basis(np.atleast_2d(r_eci), np.atleast_2d(v_eci))  # (N,3,3)
    result = np.einsum('nji,nj->ni', Q, np.atleast_2d(vecs))           # Qᵀ applied
    return result[0] if scalar else result


def azel_to_enu(
        az_rad: float,
        el_rad: float,
) -> npt.NDArray[np.floating]:
    """Unit direction vector in the ENU frame from azimuth and elevation.

    Parameters
    ----------
    az_rad : float
        Azimuth from north (rad), measured clockwise (east-positive).
    el_rad : float
        Elevation from the horizon (rad).

    Returns
    -------
    npt.NDArray[np.floating], shape (3,)
        Unit vector ``[east, north, up]``.
    """
    cos_el = np.cos(el_rad)
    return np.array([
        np.sin(az_rad) * cos_el,
        np.cos(az_rad) * cos_el,
        np.sin(el_rad),
    ], dtype=np.float64)


def enu_to_ecef(
        vecs: npt.ArrayLike,
        lat: float,
        lon: float,
) -> npt.NDArray[np.floating]:
    """Rotate vectors from local ENU (East-North-Up) to ECEF.

    Parameters
    ----------
    vecs : array_like, shape (3,) or (N, 3)
        Vectors in the ENU frame at the station location.
    lat : float
        Geodetic latitude (rad).
    lon : float
        Longitude (rad).

    Returns
    -------
    npt.NDArray[np.floating]
        Vectors in ECEF, same shape as *vecs*.
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    scalar = vecs.ndim == 1

    sl, cl = np.sin(lat), np.cos(lat)
    sn, cn = np.sin(lon), np.cos(lon)

    # Columns are E-hat, N-hat, U-hat expressed in ECEF
    R = np.array([[-sn, -sl * cn, cl * cn],
                  [ cn, -sl * sn, cl * sn],
                  [ 0.,       cl,      sl]], dtype=np.float64)  # (3, 3)

    result = (R @ np.atleast_2d(vecs).T).T  # (N, 3)
    return result[0] if scalar else result


def sun_vec_eci(
        t: np.datetime64 | npt.NDArray[np.datetime64],
) -> npt.NDArray[np.floating]:
    """Unit vector(s) pointing toward the Sun in the ECI frame.

    Uses the Astronomical Almanac low-precision solar coordinates
    (~0.01° accuracy).  The Sun's ecliptic longitude is converted to ECI
    (mean equatorial J2000) by rotating by the mean obliquity about the
    x-axis.

    Parameters
    ----------
    t : np.datetime64 | npt.NDArray[np.datetime64]
        Epoch(s) as ``datetime64[us]``.  Scalar or ``(N,)`` array.

    Returns
    -------
    npt.NDArray[np.floating]
        Unit vector(s) toward the Sun in ECI.  Shape ``(3,)`` for a
        scalar epoch, ``(N, 3)`` for an array of N epochs.
    """
    t      = np.asarray(t, dtype='datetime64[us]')
    scalar = t.ndim == 0
    t      = np.atleast_1d(t)

    d   = (t - _J2000_US).astype(np.float64) * 1e-6 / 86400.0
    L   = np.radians((280.460  + 0.9856474 * d) % 360.0)
    g   = np.radians((357.528  + 0.9856003 * d) % 360.0)
    lam = L + np.radians(1.915 * np.sin(g) + 0.020 * np.sin(2.0 * g))
    eps = np.radians(23.439 - 0.0000004 * d)

    result = np.stack([np.cos(lam),
                       np.sin(lam) * np.cos(eps),
                       np.sin(lam) * np.sin(eps)], axis=-1)  # (N, 3)
    return result[0] if scalar else result