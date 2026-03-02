"""
missiontools
============
Space mission analysis and design (SMAD) toolkit.

Submodules
----------
orbit      Orbital mechanics and propagation
power      Power budget and eclipse analysis
attitude   Attitude determination and control (ADCS)
comm       Link budget analysis
thermal    Thermal analysis
radiation  Radiation environment
coverage   Coverage and access analysis

Conventions
-----------
- All physical quantities use SI base units (m, kg, s, A, K, ...) unless
  explicitly stated otherwise in a function's docstring.
- All angles are in radians unless explicitly stated otherwise.
"""

__version__ = "0.1.0"

from .spacecraft import Spacecraft
from .attitude import AttitudeLaw
from .ground_station import GroundStation
from .aoi import AoI
from .sensor import Sensor
from .coverage_analysis import Coverage
