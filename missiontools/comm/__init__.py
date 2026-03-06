"""
missiontools.comm
=================
Link budget analysis.

Antenna classes
---------------
:class:`AbstractAntenna`
    Base class for antennas attachable to Spacecraft or GroundStation.
:class:`IsotropicAntenna`
    Constant-gain antenna (direction-independent).
:class:`SymmetricAntenna`
    Axially symmetric antenna defined by a gain-vs-angle table.

Link budget
-----------
:class:`Link`
    RF link between two antennas.  Computes link margin via
    :meth:`~Link.link_margin`.
"""

from .antenna import AbstractAntenna, IsotropicAntenna, SymmetricAntenna
from .link import Link

__all__ = ['AbstractAntenna', 'IsotropicAntenna', 'SymmetricAntenna', 'Link']
