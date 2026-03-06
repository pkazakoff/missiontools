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

Planned functionality
---------------------
- RF link budget (uplink / downlink)
- Free-space path loss
- Noise figure and system noise temperature
- Eb/N0 and link margin
"""

from .antenna import AbstractAntenna, IsotropicAntenna, SymmetricAntenna

__all__ = ['AbstractAntenna', 'IsotropicAntenna', 'SymmetricAntenna']
