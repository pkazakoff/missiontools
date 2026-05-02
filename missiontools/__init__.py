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

    from missiontools import AoI, Coverage, ConicSensor

    sensor = ConicSensor(30.0, body_vector=[0, 0, 1])
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
:class:`AbstractSensor`
    Abstract base class for instruments attached to a spacecraft.
:class:`ConicSensor`
    An instrument with a conical field of view, attached to a spacecraft
    via :meth:`~Spacecraft.add_sensor`.
:class:`RectangularSensor`
    An instrument with a rectangular (pyramidal) field of view defined by
    two half-angles, attached to a spacecraft via :meth:`~Spacecraft.add_sensor`.
:class:`AbstractAttitudeLaw`
    Abstract base class for spacecraft/sensor pointing laws.
:class:`FixedAttitudeLaw`
    Fixed-frame attitude law (LVLH, ECI, or ECEF).  Includes
    :meth:`~FixedAttitudeLaw.nadir` convenience constructor.
:class:`TrackAttitudeLaw`
    Target-tracking attitude law.  Tracks a :class:`Spacecraft` or
    :class:`GroundStation`.
:class:`CustomAttitudeLaw`
    User-supplied quaternion callback attitude law.
:class:`LimbAttitudeLaw`
    Limb-pointing attitude law.
:class:`ConditionAttitudeLaw`
    Conditional attitude law that routes between child laws based on
    a chain of :class:`AbstractCondition` predicates.
:class:`AbstractCondition`
    Abstract base class for boolean time-domain predicates.
:class:`SpaceGroundAccessCondition`
    Condition that holds when a spacecraft is above the horizon (or a
    specified minimum elevation) as seen from a ground station.
:class:`SunlightCondition`
    Condition that holds when an object (spacecraft or ground station) is
    in sunlight.
:class:`SubSatelliteRegionCondition`
    Condition that holds when a spacecraft's sub-satellite point is inside
    an :class:`AoI`.
:class:`VisibilityCondition`
    Condition that holds when two objects have unobstructed line-of-sight.
:class:`AndCondition`
    Logical AND of two conditions.  Also available via the ``&`` operator.
:class:`OrCondition`
    Logical OR of two conditions.  Also available via the ``|`` operator.
:class:`NotCondition`
    Logical NOT of a condition.  Also available via the ``~`` operator.
:class:`XorCondition`
    Logical XOR of two conditions.  Also available via the ``^`` operator.
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
:class:`ThermalCircuit`
    Lumped-parameter thermal network for transient and steady-state analysis.
:class:`NormalVectorThermalConfig`
    Surface thermal config defined by face normal vectors, areas,
    emissivities, and absorptivities.
:class:`ThermalResult`
    Container for thermal simulation results (node temperature histories).
:class:`AbstractThermalConfig`
    Base class for surface thermal configurations.

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
:mod:`~missiontools.condition`
    Boolean time-domain conditions for control logic.
:mod:`~missiontools.coverage`
    Coverage and access analysis, geographic area sampling.
:mod:`~missiontools.power`
    Solar power generation modelling.
:mod:`~missiontools.comm`
    Antenna gain and link budget analysis.
:mod:`~missiontools.thermal`
    Thermal analysis: lumped-parameter networks and surface thermal models.

"""

__version__ = "0.2.0"

__all__ = [
    "Spacecraft",
    "AbstractSensor",
    "ConicSensor",
    "RectangularSensor",
    "AbstractAttitudeLaw",
    "FixedAttitudeLaw",
    "TrackAttitudeLaw",
    "CustomAttitudeLaw",
    "LimbAttitudeLaw",
    "ConditionAttitudeLaw",
    "AbstractCondition",
    "SpaceGroundAccessCondition",
    "SunlightCondition",
    "SubSatelliteRegionCondition",
    "VisibilityCondition",
    "AndCondition",
    "OrCondition",
    "NotCondition",
    "XorCondition",
    "GroundStation",
    "AoI",
    "Coverage",
    "AbstractSolarConfig",
    "NormalVectorSolarConfig",
    "ThermalCircuit",
    "ThermalResult",
    "AbstractThermalConfig",
    "NormalVectorThermalConfig",
    "IsotropicAntenna",
    "SymmetricAntenna",
    "Link",
    "clear_cache",
    "set_cache_limit",
    "cache_info",
]

from .spacecraft import Spacecraft
from .attitude import (
    AbstractAttitudeLaw,
    FixedAttitudeLaw,
    TrackAttitudeLaw,
    CustomAttitudeLaw,
    LimbAttitudeLaw,
    ConditionAttitudeLaw,
)
from .condition import (
    AbstractCondition,
    SpaceGroundAccessCondition,
    SunlightCondition,
    SubSatelliteRegionCondition,
    VisibilityCondition,
    AndCondition,
    OrCondition,
    NotCondition,
    XorCondition,
)
from .ground_station import GroundStation
from .aoi import AoI
from .sensor import AbstractSensor, ConicSensor, RectangularSensor
from .coverage_analysis import Coverage
from .power import AbstractSolarConfig, NormalVectorSolarConfig
from .thermal import (
    ThermalCircuit,
    ThermalResult,
    AbstractThermalConfig,
    NormalVectorThermalConfig,
)
from .comm import IsotropicAntenna, SymmetricAntenna, Link
from .cache import clear_cache, set_cache_limit, cache_info
