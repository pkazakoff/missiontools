"""
missiontools.condition
=====================
Boolean time-domain conditions for use with control logic.

A :class:`Condition` evaluates to a boolean array over time, and is the
primitive building block for control logic such as
:class:`~missiontools.attitude.ConditionAttitudeLaw` (which routes between
attitude laws based on a chain of conditions).

Hierarchy
---------
:class:`AbstractCondition` (ABC)
├── :class:`SpaceGroundAccessCondition` — true when a spacecraft is
    visible from a ground station above a minimum elevation angle.
├── :class:`SunlightCondition` — true when an object is in sunlight.
├── :class:`SubSatelliteRegionCondition` — true when a spacecraft's
    sub-satellite point is inside an AoI.
├── :class:`VisibilityCondition` — true when two objects have
    unobstructed line-of-sight.
├── :class:`AndCondition` — logical AND of two conditions.
├── :class:`OrCondition` — logical OR of two conditions.
├── :class:`NotCondition` — logical NOT of a condition.
└── :class:`XorCondition` — logical XOR of two conditions.

Boolean operators ``&``, ``|``, ``^``, ``~`` are available on every
:class:`AbstractCondition` instance.
"""

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

__all__ = [
    "AbstractCondition",
    "SpaceGroundAccessCondition",
    "SunlightCondition",
    "SubSatelliteRegionCondition",
    "VisibilityCondition",
    "AndCondition",
    "OrCondition",
    "NotCondition",
    "XorCondition",
]
