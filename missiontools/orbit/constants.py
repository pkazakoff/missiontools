import numpy as np

# Standard gravitational parameters (μ = GM) in m³/s²
# Source: https://en.wikipedia.org/wiki/Standard_gravitational_parameter
# Uncertain digits (parenthetical notation) are truncated.

SUN_MU = 1.32712440042e20
MERCURY_MU = 2.2031870799e13
VENUS_MU = 3.24858592e14
EARTH_MU = 3.986004418e14
MOON_MU = 4.902800118e12
MARS_MU = 4.282837e13
CERES_MU = 6.26325e10
JUPITER_MU = 1.26686534e17
SATURN_MU = 3.7931187e16
URANUS_MU = 5.793939e15
NEPTUNE_MU = 6.836529e15
PLUTO_MU = 8.71e11
ERIS_MU = 1.108e12

# Earth J2 perturbation parameter (μ × J2 × R_E²) in m⁵/s²
# Converted from 1.75553 × 10¹⁰ km⁵/s² (× 10¹⁵ m⁵/km⁵)
EARTH_J2 = 1.75553e25

# Earth mean (arithmetic) radius in meters, as per IUGG
EARTH_MEAN_RADIUS = 6_371_008.7714

# WGS84 ellipsoid defining parameters
# Source: NGA.STND.0036_1.0.0_WGS84 (2014)
EARTH_SEMI_MAJOR_AXIS = 6_378_137.0  # m  (semi-major axis, a)
EARTH_INVERSE_FLATTENING = 298.257223563  # dimensionless (1/f)

_J2000_US = np.datetime64("2000-01-01T12:00:00", "us")
