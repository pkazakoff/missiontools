"""
missiontools.orbit
==================
Orbital mechanics, propagation, coordinate frame transformations, and access
analysis.

Propagation
-----------
:func:`propagate_analytical`
    Analytically propagate a Keplerian orbit to ECI state vectors.
    Supports unperturbed two-body motion and mean J2 secular perturbations.
:func:`sun_synchronous_inclination`
    Required inclination for a sun-synchronous RAAN precession rate.
:func:`sun_synchronous_orbit`
    Keplerian elements for a circular sun-synchronous orbit, given altitude
    and local solar time at the ascending or descending node.
:func:`geostationary_orbit`
    Keplerian elements for a geostationary orbit at a given longitude.
:func:`highly_elliptical_orbit`
    Keplerian elements for a critically inclined HEO (Molniya-style) with
    frozen apsides, given period, eccentricity, and apogee placement.

Frame transformations
---------------------
:func:`gmst`
    Greenwich Mean Sidereal Time (rad) from UTC epochs using the IAU 1982
    polynomial.
:func:`eci_to_ecef` / :func:`ecef_to_eci`
    Rotate vectors between ECI and ECEF via GMST rotation.
:func:`geodetic_to_ecef`
    Convert WGS84 geodetic coordinates (latitude, longitude, altitude) to
    ECEF Cartesian.
:func:`eci_to_lvlh` / :func:`lvlh_to_eci`
    Rotate vectors between ECI and the LVLH (RSW) frame, defined with
    **x̂** = radial, **ŷ** = along-track, **ẑ** = orbit-normal.
:func:`sun_vec_eci`
    Low-precision (≈0.01°) unit vector toward the Sun in ECI using the
    Astronomical Almanac algorithm.

Access analysis
---------------
:func:`earth_access`
    Instantaneous boolean visibility array for a ground station.
:func:`earth_access_intervals`
    Time intervals when a satellite is visible above an elevation mask.
    Uses coarse scan followed by binary-search edge refinement.
:func:`space_to_space_access`
    Instantaneous LOS check between two spacecraft (spherical body model).
:func:`space_to_space_access_intervals`
    Time intervals with unobstructed LOS between two spacecraft.

Eclipse
-------
:func:`in_sunlight`
    Cylindrical shadow model — returns ``True`` where the spacecraft is
    outside the Earth's shadow.

Planned functionality
---------------------
- Numerical orbit propagation (RK4 / RK78)
- Orbital manoeuvres (Hohmann, plane change, bi-elliptic)
- Relative motion (Clohessy–Wiltshire / Hill's equations)
- Ground track computation
"""

from .propagation import (propagate_analytical, sun_synchronous_inclination,
                          sun_synchronous_orbit, geostationary_orbit,
                          highly_elliptical_orbit)
from .frames import (gmst, eci_to_ecef, ecef_to_eci, geodetic_to_ecef,
                     eci_to_lvlh, lvlh_to_eci, sun_vec_eci,
                     azel_to_enu, enu_to_ecef)
from .access import (earth_access, earth_access_intervals,
                     space_to_space_access, space_to_space_access_intervals)
from .shadow import in_sunlight

__all__ = [
    'propagate_analytical', 'sun_synchronous_inclination', 'sun_synchronous_orbit',
    'geostationary_orbit', 'highly_elliptical_orbit',
    'gmst', 'eci_to_ecef', 'ecef_to_eci', 'geodetic_to_ecef',
    'eci_to_lvlh', 'lvlh_to_eci', 'sun_vec_eci',
    'azel_to_enu', 'enu_to_ecef',
    'earth_access', 'earth_access_intervals',
    'space_to_space_access', 'space_to_space_access_intervals',
    'in_sunlight',
]
