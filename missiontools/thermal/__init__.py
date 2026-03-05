"""
missiontools.thermal
====================
Thermal analysis.

Lumped-parameter thermal network
--------------------------------
:class:`ThermalCircuit`
    Build a network of thermal capacitances, heat sources, and active
    coolers connected by thermal resistances.  Solve transient or
    steady-state thermal response.

:class:`ThermalResult`
    Container for simulation results (time history of node temperatures).

Surface thermal models
----------------------
:class:`AbstractThermalConfig`
    Base class for surface thermal configurations.

:class:`NormalVectorThermalConfig`
    Surface thermal config defined by face normal vectors, areas,
    emissivities, and absorptivities.
"""

from .thermal_circuit import ThermalCircuit, ThermalResult
from .thermal_config import AbstractThermalConfig, NormalVectorThermalConfig

__all__ = [
    'ThermalCircuit',
    'ThermalResult',
    'AbstractThermalConfig',
    'NormalVectorThermalConfig',
]
