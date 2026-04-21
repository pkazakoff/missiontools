"""
missiontools.sensor
===================
Instrument sensor classes for spacecraft field-of-view modelling.

The base class is :class:`AbstractSensor`, with concrete subclasses for
each sensor geometry:

- :class:`ConicSensor` — sensor with a conical field of view, defined by a
  half-angle.  The boresight can be driven by an independent
  :class:`~missiontools.AbstractAttitudeLaw` or fixed in the spacecraft body
  frame.
"""

from .sensor_law import AbstractSensor, ConicSensor

__all__ = ['AbstractSensor', 'ConicSensor']
