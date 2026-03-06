"""
missiontools
============
Space mission analysis and design (SMAD) toolkit.

missiontools provides a high-level Python interface for spacecraft mission
analysis, covering orbital mechanics, coverage and access analysis, attitude
determination and control, and solar power modelling.

Quick start
-----------
Propagate a sun-synchronous LEO orbit for one day::

    import numpy as np
    from missiontools import Spacecraft

    sc = Spacecraft.sunsync(altitude_km=550, node_solar_time='10:30')
    state = sc.propagate(
        np.datetime64('2025-01-01', 'us'),
        np.datetime64('2025-01-02', 'us'),
        np.timedelta64(60, 's'),
    )
    # state['r'] — (N, 3) ECI position array (m)
    # state['v'] — (N, 3) ECI velocity array (m/s)

Compute ground coverage::

    from missiontools import AoI, Coverage, Sensor

    sensor = Sensor(30.0, body_vector=[0, 0, 1])
    sc.add_sensor(sensor)

    aoi    = AoI.from_geography('Australia')
    cov    = Coverage(aoi, [sensor])
    result = cov.coverage_fraction(
        np.datetime64('2025-01-01', 'us'),
        np.datetime64('2025-01-02', 'us'),
    )
    print(f"1-day coverage: {result['final_cumulative']:.1%}")

Classes
-------
:class:`Spacecraft`
    Defines a satellite orbit via Keplerian elements.  Factory methods
    :meth:`~Spacecraft.sunsync`, :meth:`~Spacecraft.geostationary`, and
    :meth:`~Spacecraft.heo` cover the most common orbit types.
:class:`Sensor`
    An instrument with a conical field of view, attached to a spacecraft
    via :meth:`~Spacecraft.add_sensor`.
:class:`AttitudeLaw`
    Spacecraft or sensor pointing law.  Supports nadir, fixed-frame, and
    target-tracking modes with optional yaw steering.
:class:`GroundStation`
    A ground station defined in WGS84 geodetic coordinates, with an
    :meth:`~GroundStation.access` method for contact scheduling.
:class:`AoI`
    Area of interest defined by a sampled point cloud.  Factory methods
    :meth:`~AoI.from_region`, :meth:`~AoI.from_shapefile`, and
    :meth:`~AoI.from_geography` cover rectangular regions, ESRI shapefiles,
    and Natural Earth geographies respectively.
:class:`Coverage`
    Coverage and revisit analysis for one or more sensors over an AoI.
    Supports single-satellite and constellation configurations.
:class:`AbstractSolarConfig`
    Abstract base class for solar power models.
:class:`NormalVectorSolarConfig`
    Concrete solar config defined by panel normal vectors and areas.

Conventions
-----------
- All physical quantities use SI base units (m, kg, s, A, K, ...) unless
  explicitly stated otherwise in a function's docstring.
- All angles are in radians unless explicitly stated otherwise.

Submodules
----------
:mod:`~missiontools.orbit`
    Orbital mechanics, propagation, frame transformations, and access analysis.
:mod:`~missiontools.attitude`
    Attitude law representations.
:mod:`~missiontools.coverage`
    Coverage and access analysis, geographic area sampling.
:mod:`~missiontools.power`
    Solar power generation modelling.
:mod:`~missiontools.comm`
    Antenna gain and link budget analysis.
:mod:`~missiontools.thermal`
    Thermal analysis *(planned)*.
:mod:`~missiontools.radiation`
    Radiation environment *(planned)*.
"""

__version__ = "0.1.0"

from .spacecraft import Spacecraft
from .attitude import AttitudeLaw
from .ground_station import GroundStation
from .aoi import AoI
from .sensor import Sensor
from .coverage_analysis import Coverage
from .power import AbstractSolarConfig, NormalVectorSolarConfig
from .thermal import ThermalCircuit, NormalVectorThermalConfig
from .comm import IsotropicAntenna, SymmetricAntenna
from .cache import clear_cache, set_cache_limit, cache_info
